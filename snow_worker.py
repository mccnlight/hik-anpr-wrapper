import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from combined_merger import init_merger

# === Конфиг через переменные окружения ===

SNOW_VIDEO_SOURCE_URL = os.getenv("SNOW_VIDEO_SOURCE_URL", "")
SNOW_CAMERA_ID = os.getenv("SNOW_CAMERA_ID", "camera-snow")
SNOW_YOLO_MODEL_PATH = os.getenv("SNOW_YOLO_MODEL_PATH", "yolov8n.pt")

TRUCK_CLASS_ID = int(os.getenv("SNOW_TRUCK_CLASS_ID", "7"))
CONFIDENCE_THRESHOLD = float(os.getenv("SNOW_CONFIDENCE_THRESHOLD", "0.55"))
MIN_BBOX_AREA = int(os.getenv("SNOW_MIN_BBOX_AREA", "40000"))  # Минимальная площадь bbox (px^2) для приоритета
MIN_BBOX_W = int(os.getenv("SNOW_MIN_BBOX_W", "180"))          # Минимальная ширина bbox
MIN_BBOX_H = int(os.getenv("SNOW_MIN_BBOX_H", "120"))          # Минимальная высота bbox
BBOX_EDGE_MARGIN = int(os.getenv("SNOW_BBOX_EDGE_MARGIN", "10"))  # Отступ от краев кадра, ближе — штрафуем

CENTER_ZONE_START_X = float(os.getenv("SNOW_CENTER_ZONE_START_X", "0.15"))
CENTER_ZONE_END_X = float(os.getenv("SNOW_CENTER_ZONE_END_X", "0.85"))
CENTER_ZONE_START_Y = float(os.getenv("SNOW_CENTER_ZONE_START_Y", "0.0"))  # Начало зоны по вертикали (0 = верх)
CENTER_ZONE_END_Y = float(os.getenv("SNOW_CENTER_ZONE_END_Y", "1.0"))  # Конец зоны по вертикали (1.0 = низ, весь кадр)
CENTER_LINE_X = float(os.getenv("SNOW_CENTER_LINE_X", "0.5"))
MIDDLE_ZONE_START_X = float(os.getenv("SNOW_MIDDLE_ZONE_START_X", "0.35"))  # Узкая средняя зона для триггера снимка
MIDDLE_ZONE_END_X = float(os.getenv("SNOW_MIDDLE_ZONE_END_X", "0.65"))
MIN_DIRECTION_DELTA = int(os.getenv("SNOW_MIN_DIRECTION_DELTA", "5"))
MISS_RESET_THRESHOLD_ENV = int(os.getenv("SNOW_MISS_RESET_THRESHOLD", "3"))
STATIONARY_TIMEOUT_SECONDS = float(os.getenv("SNOW_STATIONARY_TIMEOUT_SECONDS", "10.0"))  # Если машина стоит > N сек, сбрасываем трекинг
R2L_CONFIRM_THRESHOLD = int(os.getenv("SNOW_R2L_CONFIRM_THRESHOLD", "5"))  # После N подтверждений R→L игнорируем машину
STATIONARY_HARD_TIMEOUT_SECONDS = float(os.getenv("SNOW_STATIONARY_HARD_TIMEOUT_SECONDS", "60.0"))  # После 1 минуты стоянки игнорируем машину
LEAVE_RESET_THRESHOLD = int(os.getenv("SNOW_LEAVE_RESET_THRESHOLD", "12"))  # Сколько кадров подряд без детекта считать, что машина ушла

SHOW_WINDOW = os.getenv("SNOW_SHOW_WINDOW", "false").lower() == "true"

_snow_thread: threading.Thread | None = None
_stop_event = threading.Event()


# === Вспомогательные функции из старого снежного сервиса ===

def _detect_truck_bbox(frame: np.ndarray, model: YOLO) -> Optional[Tuple[int, int, int, int]]:
    """
    Находит bbox грузовика (class=TRUCK_CLASS_ID) с приоритетом ближе к камере и в центральной полосе.
    Приоритет: больший y2 (ниже в кадре) + площадь, бонус за пересечение узкой средней зоны,
    штраф за близость к краю, фильтр по минимальной площади/размеру.
    """
    best_box = None
    best_score = -1.0
    fh, fw = frame.shape[:2]

    results = model(frame, verbose=False)
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            if cls_id != TRUCK_CLASS_ID or conf < CONFIDENCE_THRESHOLD:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            w = x2 - x1
            h = y2 - y1
            area = w * h

            # Фильтр по минимальным размерам
            if area < MIN_BBOX_AREA or w < MIN_BBOX_W or h < MIN_BBOX_H:
                continue

            # Пересечение узкой средней зоны по X (используем глобальные коэффициенты)
            mid_start = int(fw * MIDDLE_ZONE_START_X)
            mid_end = int(fw * MIDDLE_ZONE_END_X)
            overlaps_mid = not (x2 < mid_start or x1 > mid_end)

            # Бонус/штраф
            score = y2 * 1.0 + area * 0.0005
            if overlaps_mid:
                score += 1000.0
            # Штраф за близость к краям кадра
            if x1 <= BBOX_EDGE_MARGIN or x2 >= fw - BBOX_EDGE_MARGIN or y1 <= BBOX_EDGE_MARGIN or y2 >= fh - BBOX_EDGE_MARGIN:
                score -= 1500.0

            if score > best_score:
                best_score = score
                best_box = (x1, y1, x2, y2)
    return best_box


def _check_center_zone(bbox, frame_width: int, frame_height: int):
    """
    Проверка: bbox пересекается с центральной зоной по X и Y (не только центр).
    Это позволяет захватывать машины, которые едут близко к краю зоны, включая нижнюю часть кадра.
    """
    # Небольшое сужение зоны от краёв кадра, чтобы не реагировать на объекты вне сцены
    TRIM_MARGIN = 0.02  # 2% от кадра по каждой стороне
    x1, y1, x2, y2 = bbox
    center_x = x1 + (x2 - x1) // 2
    center_y = y1 + (y2 - y1) // 2
    
    # Зона по горизонтали (X)
    zone_start_px = int(frame_width * CENTER_ZONE_START_X)
    zone_end_px = int(frame_width * CENTER_ZONE_END_X)
    
    # Зона по вертикали (Y) - теперь учитываем нижнюю часть кадра
    zone_start_py = int(frame_height * CENTER_ZONE_START_Y)
    zone_end_py = int(frame_height * CENTER_ZONE_END_Y)

    # Дополнительно отступаем от краёв кадра на margin, чтобы не ловить почти вышедшие за кадр объекты
    margin_x = int(frame_width * TRIM_MARGIN)
    margin_y = int(frame_height * TRIM_MARGIN)
    zone_start_px = max(zone_start_px, margin_x)
    zone_end_px = min(zone_end_px, frame_width - margin_x)
    zone_start_py = max(zone_start_py, margin_y)
    zone_end_py = min(zone_end_py, frame_height - margin_y)
    
    # Проверяем пересечение bbox с зоной по обеим осям
    # Грузовик считается в зоне, если хотя бы часть его bbox пересекается с зоной
    bbox_in_zone_x = (x1 < zone_end_px) and (x2 > zone_start_px)
    bbox_in_zone_y = (y1 < zone_end_py) and (y2 > zone_start_py)
    bbox_in_zone = bbox_in_zone_x and bbox_in_zone_y
    
    return bbox_in_zone, center_x, center_y, zone_start_px, zone_end_px, zone_start_py, zone_end_py


def _is_moving_left_to_right(current_center_x: int, last_center_x: Optional[int]) -> bool:
    if last_center_x is None:
        return False
    return (current_center_x - last_center_x) > MIN_DIRECTION_DELTA


def _encode_frame_to_jpeg(frame: np.ndarray) -> Tuple[bytes, datetime]:
    """
    Превратить кадр в JPEG-байты без записи на диск.
    """
    ts = datetime.now(tz=timezone.utc)
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("cannot encode frame to JPEG")
    return buf.tobytes(), ts


# === Основной цикл снежной камеры ===

def _snow_loop(upstream_url: str):
    model = YOLO(SNOW_YOLO_MODEL_PATH)
    merger = init_merger(upstream_url)

    cap = cv2.VideoCapture(SNOW_VIDEO_SOURCE_URL)
    if not cap.isOpened():
        print(f"[SNOW] cannot open video source: {SNOW_VIDEO_SOURCE_URL}")
        return

    window_name = "Snow Camera" if SHOW_WINDOW else None
    if SHOW_WINDOW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

    last_center_x = None
    event_sent_for_current_truck = False
    last_movement_time = None  # Время последнего значимого движения
    last_truck_bbox = None  # Последний bbox для отслеживания смены машины
    last_truck_was_r_to_l = False  # Флаг: последняя машина двигалась справа налево (сохраняется между кадрами)
    r2l_confirmations = 0  # Сколько раз подряд подтвердили движение R→L для текущей машины
    ignore_current_truck = False  # Флаг: игнорировать текущую машину (R→L подтверждена или стоит слишком долго)

    frame_width = None
    frame_height = None
    center_start_px = None
    center_end_px = None
    center_start_py = None
    center_end_py = None
    center_x_geom = None

    fail_count = 0
    MAX_FAILS = 50
    frame_count = 0
    miss_count = 0          # Счетчик подряд идущих кадров без детекции
    MISS_RESET_THRESHOLD = MISS_RESET_THRESHOLD_ENV  # После скольких пропусков сбрасывать трекинг
    leave_count = 0         # Счетчик подряд идущих кадров без детекции для определения, что машина ушла

    print("[SNOW] worker started")
    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            fail_count += 1
            print(f"[SNOW] read fail {fail_count}")
            if fail_count >= MAX_FAILS:
                print("[SNOW] reopening stream...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(SNOW_VIDEO_SOURCE_URL)
                fail_count = 0
            time.sleep(0.05)
            continue

        fail_count = 0
        frame_count += 1
        raw_frame = frame.copy()

        if frame_width is None:
            frame_height, frame_width = frame.shape[:2]
            center_start_px = int(frame_width * CENTER_ZONE_START_X)
            center_end_px = int(frame_width * CENTER_ZONE_END_X)
            center_start_py = int(frame_height * CENTER_ZONE_START_Y)
            center_end_py = int(frame_height * CENTER_ZONE_END_Y)
            center_x_geom = int(frame_width * CENTER_LINE_X)
            print(f"[SNOW] center zone X: {center_start_px}px .. {center_end_px}px (width: {frame_width}px)")
            print(f"[SNOW] center zone Y: {center_start_py}px .. {center_end_py}px (height: {frame_height}px)")

        # Отрисовка вспомогательных линий (только для визуального контроля)
        if SHOW_WINDOW:
            cv2.line(frame, (center_x_geom, 0), (center_x_geom, frame_height), (0, 255, 255), 1)
            cv2.line(frame, (center_start_px, 0), (center_start_px, frame_height), (0, 255, 0), 2)
            cv2.line(frame, (center_end_px, 0), (center_end_px, frame_height), (0, 255, 0), 2)
            cv2.line(frame, (0, center_start_py), (frame_width, center_start_py), (0, 255, 0), 2)
            cv2.line(frame, (0, center_end_py), (frame_width, center_end_py), (0, 255, 0), 2)

        bbox = _detect_truck_bbox(raw_frame, model)
        if bbox:
            leave_count = 0  # Есть детекция — сбрасываем счетчик ухода
            in_zone, center_x_obj, center_y_obj, zone_start_px, zone_end_px, zone_start_py, zone_end_py = _check_center_zone(bbox, frame_width, frame_height)
            
            # Проверяем, не сменилась ли машина (по значительному изменению bbox)
            is_new_truck = False
            if last_truck_bbox is not None:
                # Вычисляем IoU (Intersection over Union) или просто расстояние между центрами
                last_x1, last_y1, last_x2, last_y2 = last_truck_bbox
                last_center_x_bbox = (last_x1 + last_x2) // 2
                last_center_y_bbox = (last_y1 + last_y2) // 2
                current_center_x_bbox = (bbox[0] + bbox[2]) // 2
                current_center_y_bbox = (bbox[1] + bbox[3]) // 2
                
                # Если центр сместился больше чем на 30% размера кадра, это новая машина
                distance_threshold = max(frame_width, frame_height) * 0.3
                center_distance = ((current_center_x_bbox - last_center_x_bbox) ** 2 + 
                                 (current_center_y_bbox - last_center_y_bbox) ** 2) ** 0.5
                if center_distance > distance_threshold:
                    is_new_truck = True
                    print(f"[SNOW] new truck detected (center distance={center_distance:.1f}px > {distance_threshold:.1f}px), resetting tracking")
                    last_center_x = None
                    event_sent_for_current_truck = False
                    last_movement_time = None
                    last_truck_was_r_to_l = False  # Новая машина - сбрасываем флаг R→L
                    r2l_confirmations = 0
                    ignore_current_truck = False
            
            # Логируем состояние для диагностики (только при детекции)
            # Не логируем если машина стоит слишком долго (чтобы не спамить)
            should_log_detection = True
            if last_movement_time is not None:
                time_since_movement = time.time() - last_movement_time
                if time_since_movement > STATIONARY_TIMEOUT_SECONDS:
                    should_log_detection = False  # Не логируем стоячие машины
            
            if should_log_detection:
                x1, y1, x2, y2 = bbox
                print(f"[SNOW] TRUCK DETECTED: bbox=({x1},{y1},{x2},{y2}), in_zone={in_zone}, "
                      f"center=({center_x_obj:.1f},{center_y_obj:.1f})px, "
                      f"zone_x={zone_start_px}-{zone_end_px}px, zone_y={zone_start_py}-{zone_end_py}px, "
                      f"last_center_x={last_center_x}, event_sent={event_sent_for_current_truck}, "
                      f"is_new_truck={is_new_truck}")
            
            # Проверяем направление движения
            moving_right = _is_moving_left_to_right(center_x_obj, last_center_x)
            
            # Отслеживаем, было ли движение справа налево в текущем кадре.
            # При множественных подтверждениях R→L игнорируем эту машину, пока она не пропадет из кадра.
            current_frame_r_to_l = False
            if last_center_x is not None:
                delta = center_x_obj - last_center_x
                if delta < -MIN_DIRECTION_DELTA:  # Движение справа налево
                    current_frame_r_to_l = True
                    last_truck_was_r_to_l = True  # Сохраняем флаг для следующих кадров
                    r2l_confirmations += 1
                    if r2l_confirmations >= R2L_CONFIRM_THRESHOLD:
                        ignore_current_truck = True
                        print(f"[SNOW] confirmed R→L {r2l_confirmations} times (>= {R2L_CONFIRM_THRESHOLD}), ignoring this truck until it leaves")
                elif delta > MIN_DIRECTION_DELTA:
                    # Движение в нужную сторону сбрасывает подтверждения R→L
                    r2l_confirmations = 0
            
            # Сохраняем флаг первого обнаружения ДО установки last_center_x
            is_first_detection = (last_center_x is None and in_zone)
            is_first_in_left_half = (is_first_detection and 
                                     center_x_obj < (zone_start_px + zone_end_px) // 2)
            
            # Если это первое обнаружение в зоне - сохраняем позицию для следующего кадра.
            # ВАЖНО: не сбрасываем флаг R→L на первом кадре, чтобы не пропускать
            # последовательное движение справа-налево.
            if is_first_detection:
                # Считаем, что это новое появление в кадре — сбрасываем игнор после R→L
                if ignore_current_truck:
                    print("[SNOW] reset ignore flag for new truck (first detection in zone)")
                ignore_current_truck = False
                r2l_confirmations = 0
                last_truck_was_r_to_l = False
                last_center_x = center_x_obj
                last_movement_time = time.time()  # Фиксируем время первого обнаружения
                print(f"[SNOW] first detection in zone (center_x={center_x_obj:.1f}px), saved for direction check")
            # Если грузовик движется слева направо - обновляем позицию и время движения
            elif moving_right:
                last_center_x = center_x_obj
                last_movement_time = time.time()  # Обновляем время последнего движения
                r2l_confirmations = 0
                if ignore_current_truck:
                    print(f"[SNOW] truck now moving left-to-right, stopping ignore (center_x={center_x_obj:.1f}px)")
                ignore_current_truck = False
                # Если машина движется L→R, сбрасываем флаг R→L (подтверждение правильного направления)
                if last_truck_was_r_to_l:
                    print(f"[SNOW] truck moving left-to-right (center_x={center_x_obj:.1f}px), resetting R→L flag, tracking updated")
                    last_truck_was_r_to_l = False
                else:
                    print(f"[SNOW] truck moving left-to-right (center_x={center_x_obj:.1f}px), tracking updated")
            # Если грузовик движется справа налево - сбрасываем отслеживание
            # Это уезжающие машины, их нужно игнорировать
            elif last_center_x is not None:
                delta = center_x_obj - last_center_x
                if delta < -MIN_DIRECTION_DELTA:  # Движение справа налево (уезжающая машина)
                    print(f"[SNOW] FILTERED: truck moving right-to-left (delta={delta:.1f}px) - exiting vehicle, resetting tracking, setting R→L flag")
                    last_center_x = None
                    event_sent_for_current_truck = False
                    last_movement_time = None
                    last_truck_was_r_to_l = True  # Сохраняем флаг R→L для следующих кадров
                    ignore_current_truck = True  # Явно игнорируем уезжающую машину
                elif abs(delta) <= MIN_DIRECTION_DELTA:
                    # Грузовик стоит на месте или движется очень медленно
                    print(f"[SNOW] truck stationary or slow (delta={delta:.1f}px), keeping position")
                    # Не обновляем last_movement_time - машина стоит

            # Всегда рисуем квадратик на frame для логирования (даже если окно не показывается)
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # Показываем направление движения
            if last_center_x is not None:
                if moving_right:
                    cv2.putText(frame, "->", (x2 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                elif center_x_obj < last_center_x - MIN_DIRECTION_DELTA:
                    cv2.putText(frame, "<-", (x2 + 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if SHOW_WINDOW:
                # Дополнительная визуализация только если окно включено
                pass

            # Сохраняем снапшот и кладем в очередь без анализа, если:
            # 1. Грузовик в зоне
            # 2. Движется слева направо (подтверждено) ИЛИ это первое обнаружение в зоне
            # 3. Еще не отправлено событие для этого грузовика
            # 4. Машина не стоит слишком долго (если не первое обнаружение)
            # 5. НЕ движется справа налево в текущем кадре
            in_middle_zone = center_start_px + int((center_end_px - center_start_px) * (MIDDLE_ZONE_START_X - CENTER_ZONE_START_X) / (CENTER_ZONE_END_X - CENTER_ZONE_START_X)) <= center_x_obj <= \
                              center_start_px + int((center_end_px - center_start_px) * (MIDDLE_ZONE_END_X - CENTER_ZONE_START_X) / (CENTER_ZONE_END_X - CENTER_ZONE_START_X))
            
            # Проверяем, не стоит ли машина слишком долго
            should_process_truck = True
            if not is_first_detection and last_movement_time is not None:
                time_since_movement = time.time() - last_movement_time
                if time_since_movement > STATIONARY_HARD_TIMEOUT_SECONDS:
                    print(f"[SNOW] truck stationary too long ({time_since_movement:.1f}s > {STATIONARY_HARD_TIMEOUT_SECONDS}s), ignoring this truck until it leaves")
                    ignore_current_truck = True
                    last_center_x = None
                    event_sent_for_current_truck = False
                    last_movement_time = None
                    last_truck_bbox = None
                    last_truck_was_r_to_l = False
                    r2l_confirmations = 0
                    should_process_truck = False
                elif time_since_movement > STATIONARY_TIMEOUT_SECONDS:
                    # Сбрасываем трекинг для стоячей машины, чтобы не логировать постоянно
                    # и чтобы система могла начать отслеживать новую машину
                    print(f"[SNOW] truck stationary too long ({time_since_movement:.1f}s > {STATIONARY_TIMEOUT_SECONDS}s), resetting tracking")
                    last_center_x = None
                    event_sent_for_current_truck = False
                    last_movement_time = None
                    last_truck_bbox = None
                    last_truck_was_r_to_l = False  # Стоячая машина - сбрасываем флаг R→L
                    r2l_confirmations = 0
                    should_process_truck = False  # Пропускаем дальнейшую обработку для этой стоячей машины
            
            # Обрабатываем только если машина не стоит слишком долго
            if should_process_truck:
                # Упрощенные условия для добавления события:
                # 1. Машина в зоне
                # 2. Еще не отправлено событие для этого грузовика
                # 3. Движется слева направо ИЛИ это первое обнаружение в зоне
                # 4. НЕ движется справа налево в текущем кадре
                # 5. НЕ игнорируется (не было подтвержденного движения R→L)
                # Убрали требование in_middle_zone для упрощения - если машина в зоне и движется L→R, добавляем
                should_add_event = (
                    in_zone
                    and not event_sent_for_current_truck
                    and (moving_right or is_first_detection)  # Упростили: движется L→R ИЛИ первое обнаружение
                    and not current_frame_r_to_l
                    and not ignore_current_truck
                )
                
                # Подробное логирование для диагностики (всегда логируем, если машина в зоне и событие еще не отправлено)
                if in_zone and not event_sent_for_current_truck:
                    print(
                        "[SNOW] DEBUG: in_zone=True, event_sent=False, "
                        f"moving_right={moving_right}, is_first_detection={is_first_detection}, "
                        f"current_frame_r_to_l={current_frame_r_to_l}, "
                        f"ignore_current_truck={ignore_current_truck}, in_middle_zone={in_middle_zone}, "
                        f"center_x={center_x_obj:.1f}px, center_x_geom={center_x_geom}px, "
                        f"should_add_event={should_add_event}"
                    )
                    if not should_add_event:
                        # Детальная диагностика, почему событие не добавляется
                        reasons = []
                        if not in_zone:
                            reasons.append("not in zone")
                        if event_sent_for_current_truck:
                            reasons.append("event already sent")
                        if not (moving_right or is_first_detection):
                            reasons.append(f"not moving right and not first detection (moving_right={moving_right}, is_first_detection={is_first_detection})")
                        if current_frame_r_to_l:
                            reasons.append("current frame R→L")
                        if ignore_current_truck:
                            reasons.append("truck ignored")
                        print(f"[SNOW] DEBUG: event NOT added, reasons: {', '.join(reasons) if reasons else 'unknown'}")
                
                if should_add_event:
                    print(f"[SNOW] ===== ENCODING SNAPSHOT AND QUEUING (IN-MEMORY) ======")
                    try:
                        photo_bytes, ts_saved = _encode_frame_to_jpeg(raw_frame)
                    except Exception as e:
                        print(f"[SNOW] cannot encode frame to JPEG: {e}")
                        photo_bytes = None
                        ts_saved = datetime.now(tz=timezone.utc)

                    # Формируем время события в RFC3339 (ISO8601) с суффиксом Z
                    event_time_iso = ts_saved.replace(microsecond=0).isoformat()
                    if event_time_iso.endswith("+00:00"):
                        event_time_iso = event_time_iso[:-6] + "Z"
                    elif "+" in event_time_iso or "-" in event_time_iso[-6:]:
                        pass
                    else:
                        event_time_iso += "Z"
                    
                    payload = {
                        "camera_id": SNOW_CAMERA_ID,
                        "event_time": event_time_iso,
                        "bbox": list(bbox) if bbox else None,
                    }
                    print(f"[SNOW] payload queued (no Gemini yet): {payload}")

                    merger.add_snow_event(payload, photo_bytes)
                    print(f"[SNOW] snow event added to queue at {ts_saved.isoformat()}, "
                          f"photo_size={len(photo_bytes) if photo_bytes else 0} bytes, "
                          f"queue_size should increase")
                    event_sent_for_current_truck = True
                    last_truck_bbox = bbox  # Сохраняем bbox для отслеживания смены машины
                elif in_zone and not moving_right and last_center_x is not None and not event_sent_for_current_truck:
                    # Грузовик в зоне, но направление еще не подтверждено
                    # Логируем только если прошло меньше 2 секунд с последнего движения (чтобы не спамить)
                    if last_movement_time is None or (time.time() - last_movement_time) < 2.0:
                        print(f"[SNOW] truck in zone, waiting for direction confirmation (center_x={center_x_obj:.1f}px, last={last_center_x:.1f}px)")
                    last_truck_bbox = bbox  # Обновляем bbox даже если событие не добавлено
            elif not in_zone:
                # Грузовик детектирован, но не в зоне - сбрасываем состояние для следующей машины
                print(f"[SNOW] truck detected but NOT in zone: center=({center_x_obj:.1f},{center_y_obj:.1f}), "
                      f"zone_x={zone_start_px}-{zone_end_px}, zone_y={zone_start_py}-{zone_end_py}, "
                      f"bbox=({x1},{y1},{x2},{y2}), resetting tracking for next truck")
                # Сбрасываем состояние, чтобы следующая машина могла быть детектирована
                event_sent_for_current_truck = False
                last_center_x = None
                last_movement_time = None
                last_truck_bbox = None
                last_truck_was_r_to_l = False  # Машина покинула зону - сбрасываем флаг R→L
                r2l_confirmations = 0
                ignore_current_truck = False
        else:
            # Грузовик не детектирован - даем небольшой допуск на пропуски
            miss_count += 1
            leave_count += 1

            # Если подряд много пропусков, считаем что машина ушла из кадра и разрешаем новые события
            if leave_count >= LEAVE_RESET_THRESHOLD:
                if last_truck_bbox is not None or last_center_x is not None or event_sent_for_current_truck:
                    print(f"[SNOW] truck likely left scene (no detect for {leave_count} frames), resetting tracking for next truck")
                event_sent_for_current_truck = False
                last_center_x = None
                last_movement_time = None
                last_truck_bbox = None
                last_truck_was_r_to_l = False
                r2l_confirmations = 0
                ignore_current_truck = False
                leave_count = 0
                miss_count = 0

            # Небольшая пауза, чтобы не крутить цикл слишком быстро при пропусках
            time.sleep(0.02)

        if SHOW_WINDOW:
            cv2.imshow(window_name, cv2.resize(frame, (960, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                _stop_event.set()
                break

        # небольшая пауза, чтобы не грузить CPU
        time.sleep(0.005)

    cap.release()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("[SNOW] worker stopped")


def start_snow_worker(upstream_url: str):
    """
    Запуск снегового воркера в отдельном потоке.
    """
    global _snow_thread
    if _snow_thread is not None:
        return
    if not SNOW_VIDEO_SOURCE_URL:
        print("[SNOW] SNOW_VIDEO_SOURCE_URL is empty, snow worker disabled")
        return

    _stop_event.clear()
    _snow_thread = threading.Thread(
        target=_snow_loop,
        args=(upstream_url,),
        daemon=True,
        name="snow-worker",
    )
    _snow_thread.start()


def stop_snow_worker():
    """
    Остановка снегового воркера (используется только если нужно мягко завершить).
    """
    if _snow_thread is None:
        return
    _stop_event.set()
