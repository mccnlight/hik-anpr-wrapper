import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

# Убрали YOLO и combined_merger - больше не нужны для автоматических событий

# === Конфиг через переменные окружения ===

SNOW_VIDEO_SOURCE_URL = os.getenv("SNOW_VIDEO_SOURCE_URL", "")
SNOW_CAMERA_ID = os.getenv("SNOW_CAMERA_ID", "camera-snow")
SNOW_CAPTURE_DELAY_SECONDS = float(os.getenv("SNOW_CAPTURE_DELAY_SECONDS", "1.0"))  # Задержка перед захватом фото

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
SNOW_ALLOW_R2L_EVENT = os.getenv("SNOW_ALLOW_R2L_EVENT", "false").lower() == "true"  # Разрешать событие даже при движении R→L

SHOW_WINDOW = os.getenv("SNOW_SHOW_WINDOW", "false").lower() == "true"

# Принудительно настраиваем FFMPEG backend: TCP, таймаут ~5с, небольшой буфер, тихий лог FFmpeg
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;5000000|buffer_size;1024000|loglevel;quiet",
)

def _silence_opencv_logs() -> None:
    """Глушим логи OpenCV/FFmpeg; учитываем разные версии OpenCV."""
    try:
        # Новый API (OpenCV >=4.5)
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
        return
    except Exception:
        pass
    try:
        # Старый API (cv2.setLogLevel)
        cv2.setLogLevel(cv2.LOG_LEVEL_SILENT)
    except Exception:
        # Если нет ни одного API — просто продолжаем
        pass

_silence_opencv_logs()

_snow_thread: threading.Thread | None = None
_stop_event = threading.Event()
_cap: cv2.VideoCapture | None = None
_cap_lock = threading.Lock()

# Буфер кадров с временными метками для доступа к прошлым кадрам
@dataclass
class TimestampedFrame:
    frame: np.ndarray
    timestamp: float  # time.time()

_frame_buffer: deque[TimestampedFrame] = deque(maxlen=300)  # ~10 секунд при 30 FPS
_frame_buffer_lock = threading.Lock()
BUFFER_DURATION_SECONDS = 10.0  # Храним кадры за последние 10 секунд (для задержки до 9.0 сек)


# === Функция захвата фото по запросу ===

def capture_snow_photo() -> Optional[bytes]:
    """
    Захватывает кадр из буфера снеговой камеры, который был записан примерно 2.5-3 секунды назад.
    Камера по снегу стоит раньше камеры по номерам, поэтому нужно брать кадр из прошлого.
    
    Returns:
        JPEG bytes кадра или None, если не удалось найти подходящий кадр
    """
    global _frame_buffer
    import time
    
    current_time = time.time()
    target_timestamp = current_time - SNOW_CAPTURE_DELAY_SECONDS
    
    with _frame_buffer_lock:
        if len(_frame_buffer) == 0:
            print(f"[SNOW] capture_snow_photo: buffer is empty")
            return None
        
        # Ищем кадр, который ближе всего к целевому времени (2.5 сек назад)
        best_frame = None
        best_delta = float('inf')
        
        for timestamped_frame in _frame_buffer:
            delta = abs(timestamped_frame.timestamp - target_timestamp)
            if delta < best_delta:
                best_delta = delta
                best_frame = timestamped_frame
        
        if best_frame is None:
            print(f"[SNOW] capture_snow_photo: no suitable frame found")
            return None
        
        # Проверяем, что кадр не слишком старый (больше 5 секунд - уже неактуален)
        frame_age = current_time - best_frame.timestamp
        if frame_age > BUFFER_DURATION_SECONDS:
            print(f"[SNOW] capture_snow_photo: frame too old ({frame_age:.2f}s > {BUFFER_DURATION_SECONDS}s)")
            return None
        
        # Логируем для отладки
        actual_delay = frame_age
        print(f"[SNOW] capture_snow_photo: found frame from {actual_delay:.2f}s ago (requested {SNOW_CAPTURE_DELAY_SECONDS:.2f}s, delta={best_delta:.3f}s)")
        
        frame = best_frame.frame.copy()
    
    # Кодируем кадр в JPEG
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return None
        photo_bytes = buf.tobytes()
        print(f"[SNOW] capture_snow_photo: encoded frame, size={len(photo_bytes)} bytes")
        return photo_bytes
    except Exception as e:
        print(f"[SNOW] capture_snow_photo: error encoding frame: {e}")
        return None


# === Старые вспомогательные функции (оставлены для возможного будущего использования) ===

def _detect_truck_bbox(frame: np.ndarray, model) -> Optional[Tuple[int, int, int, int]]:
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


# === Основной цикл снежной камеры (упрощенный - только чтение потока) ===

def _snow_loop(upstream_url: str):
    """
    Упрощенный цикл: читает RTSP поток и сохраняет кадры в буфер с временными метками.
    Автоматические события больше не создаются - фото делается по запросу через capture_snow_photo().
    Кадры сохраняются в буфер, чтобы можно было получить кадр из прошлого (когда машина была под снеговой камерой).
    """
    global _cap, _frame_buffer
    
    _cap = cv2.VideoCapture(SNOW_VIDEO_SOURCE_URL, cv2.CAP_FFMPEG)
    if not _cap.isOpened():
        print(f"[SNOW] cannot open video source: {SNOW_VIDEO_SOURCE_URL}")
        return

    window_name = "Snow Camera" if SHOW_WINDOW else None
    if SHOW_WINDOW:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 960, 540)

    fail_count = 0
    MAX_FAILS = 15
    last_log_time = 0.0
    LOG_INTERVAL_SECONDS = 8.0  # Логируем раз в 8 секунд

    print(f"[SNOW] worker started (buffering mode - storing frames for {BUFFER_DURATION_SECONDS}s)")
    
    while not _stop_event.is_set():
        with _cap_lock:
            ret, frame = _cap.read()
        
        if not ret or frame is None or frame.size == 0:
            fail_count += 1
            if fail_count >= MAX_FAILS:
                with _cap_lock:
                    _cap.release()
                    time.sleep(2)
                    _cap = cv2.VideoCapture(SNOW_VIDEO_SOURCE_URL, cv2.CAP_FFMPEG)
                fail_count = 0
            time.sleep(0.05)
            continue

        fail_count = 0
        
        # Сохраняем кадр в буфер с временной меткой
        current_timestamp = time.time()
        timestamped_frame = TimestampedFrame(frame=frame.copy(), timestamp=current_timestamp)
        
        with _frame_buffer_lock:
            # Удаляем старые кадры (старше BUFFER_DURATION_SECONDS)
            while len(_frame_buffer) > 0:
                oldest = _frame_buffer[0]
                if current_timestamp - oldest.timestamp > BUFFER_DURATION_SECONDS:
                    _frame_buffer.popleft()
                else:
                    break
            
            # Добавляем новый кадр
            _frame_buffer.append(timestamped_frame)
            
            # Логируем раз в 8 секунд
            if current_timestamp - last_log_time >= LOG_INTERVAL_SECONDS:
                if len(_frame_buffer) > 0:
                    print(f"[SNOW] buffer size: {len(_frame_buffer)} frames, "
                          f"oldest: {current_timestamp - _frame_buffer[0].timestamp:.2f}s ago, "
                          f"newest: {current_timestamp - _frame_buffer[-1].timestamp:.2f}s ago")
                last_log_time = current_timestamp

        if SHOW_WINDOW:
            cv2.imshow(window_name, cv2.resize(frame, (960, 540)))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                _stop_event.set()
                break

        # Небольшая пауза, чтобы не грузить CPU
        time.sleep(0.01)

    with _cap_lock:
        if _cap is not None:
            _cap.release()
            _cap = None
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("[SNOW] worker stopped")


def start_snow_worker(upstream_url: str):
    """
    Запуск снегового воркера в отдельном потоке.
    Воркер читает RTSP поток и поддерживает его активным для мгновенного захвата кадров по запросу.
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
