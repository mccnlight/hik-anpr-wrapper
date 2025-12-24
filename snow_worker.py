import os
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict
from collections import deque
from dataclasses import dataclass
import concurrent.futures
import hashlib

import cv2
import numpy as np

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

_frame_buffer: deque[TimestampedFrame] = deque(maxlen=90)  # ~3 секунды при 30 FPS (уменьшено с 300 для экономии памяти)
_frame_buffer_lock = threading.Lock()
BUFFER_DURATION_SECONDS = 3.0  # Храним кадры за последние 3 секунды (уменьшено с 10.0 для экономии памяти)

# YOLO модель для детекции грузовиков
_yolo_model = None
_yolo_model_lock = threading.Lock()

# Кэш результатов детекции (timestamp -> (score, bbox))
_detection_cache: Dict[float, Tuple[float, Optional[Tuple[int, int, int, int]]]] = {}
_detection_cache_lock = threading.Lock()
DETECTION_CACHE_TTL = 30.0  # Кэш живет 30 секунд


# === Функция захвата фото по запросу ===

def _get_yolo_model():
    """Ленивая инициализация YOLO модели для детекции грузовиков"""
    global _yolo_model
    with _yolo_model_lock:
        if _yolo_model is None:
            try:
                from ultralytics import YOLO
                model_path = os.getenv("SNOW_YOLO_MODEL_PATH", "yolov8n.pt")
                _yolo_model = YOLO(model_path)
                print(f"[SNOW] YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"[SNOW] ERROR: Failed to load YOLO model: {e}")
                return None
    return _yolo_model

def _detect_all_vehicles(frame: np.ndarray, model) -> list:
    """
    Детектирует ВСЕ машины в кадре и возвращает список с информацией о каждой.
    """
    if model is None:
        return []
    
    fh, fw = frame.shape[:2]
    all_vehicles = []
    
    try:
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
                
                all_vehicles.append({
                    'bbox': (x1, y1, x2, y2),
                    'area': area,
                    'size_ratio': area / (fw * fh),
                    'center_x': (x1 + x2) / 2.0,
                    'center_y': (y1 + y2) / 2.0,
                    'conf': conf
                })
    except Exception as e:
        print(f"[SNOW] Error in YOLO detection: {e}")
        return []
    
    return all_vehicles

def _assess_frame_quality(frame: np.ndarray, timestamp: float, model) -> Tuple[float, Optional[Tuple[int, int, int, int]], int, list]:
    """
    Оценивает качество кадра: есть ли машина в центре на ближней полосе.
    Возвращает: (score, bbox, vehicles_count, all_vehicles) где score > 0 если машина найдена
    all_vehicles - список всех найденных машин для сравнения между кадрами
    """
    # Проверяем кэш
    with _detection_cache_lock:
        if timestamp in _detection_cache:
            cached_result = _detection_cache[timestamp]
            # Если в кэше старая версия, возвращаем с пустым списком
            if len(cached_result) == 3:
                return (*cached_result, [])
            return cached_result
    
    fh, fw = frame.shape[:2]
    
    # Детектируем ВСЕ грузовики в кадре
    all_vehicles = _detect_all_vehicles(frame, model)
    vehicles_count = len(all_vehicles)
    
    if vehicles_count == 0:
        result = (0.0, None, 0, [])
        with _detection_cache_lock:
            _detection_cache[timestamp] = result
        return result
    
    # Разделяем машины на ближние и дальние
    MIN_SIZE_RATIO_NEAR = 0.08  # Минимум 8% для ближней полосы
    MIN_SIZE_RATIO_FAR = 0.03   # Минимум 3% для дальней полосы (но приоритет ближним)
    
    near_lane_vehicles = []
    far_lane_vehicles = []
    
    for vehicle in all_vehicles:
        size_ratio = vehicle['size_ratio']
        bbox = vehicle['bbox']
        
        # Проверяем, что машина в центральной зоне
        in_zone, _, _, _, _, _, _ = _check_center_zone(bbox, fw, fh)
        if not in_zone:
            continue
        
        if size_ratio >= MIN_SIZE_RATIO_NEAR:
            near_lane_vehicles.append(vehicle)
        elif size_ratio >= MIN_SIZE_RATIO_FAR:
            far_lane_vehicles.append(vehicle)
    
    # Приоритет 1: ближняя полоса (большие машины)
    best_vehicle = None
    best_score = -1.0
    
    for vehicle in near_lane_vehicles:
        center_score_x = 1.0 - abs((vehicle['center_x'] / fw) - 0.5) * 2.0
        center_score_y = 1.0 - abs((vehicle['center_y'] / fh) - 0.6) * 2.0
        # Размер - главный критерий (90% веса)
        quality_score = vehicle['size_ratio'] * 900.0 + center_score_x * 5.0 + center_score_y * 5.0
        
        if quality_score > best_score:
            best_score = quality_score
            best_vehicle = vehicle
    
    # Приоритет 2: если нет ближних, берем дальние
    if best_vehicle is None and far_lane_vehicles:
        for vehicle in far_lane_vehicles:
            center_score_x = 1.0 - abs((vehicle['center_x'] / fw) - 0.5) * 2.0
            center_score_y = 1.0 - abs((vehicle['center_y'] / fh) - 0.6) * 2.0
            quality_score = vehicle['size_ratio'] * 100.0 + center_score_x * 5.0 + center_score_y * 5.0
            
            if quality_score > best_score:
                best_score = quality_score
                best_vehicle = vehicle
    
    if best_vehicle is None:
        result = (0.0, None, vehicles_count, all_vehicles)
        with _detection_cache_lock:
            _detection_cache[timestamp] = result
        return result
    
    result = (best_score, best_vehicle['bbox'], vehicles_count, all_vehicles)
    with _detection_cache_lock:
        _detection_cache[timestamp] = result
    return result

def _process_frame_for_delay(target_delay: float, current_time: float, model) -> Optional[Tuple[float, TimestampedFrame, Tuple[int, int, int, int], int, list]]:
    """Обрабатывает один кадр для заданной задержки"""
    target_timestamp = current_time - target_delay
    
    with _frame_buffer_lock:
        if len(_frame_buffer) == 0:
            return None
        
        # Ищем ближайший кадр к целевому времени
        best_frame = None
        best_delta = float('inf')
        
        for timestamped_frame in _frame_buffer:
            delta = abs(timestamped_frame.timestamp - target_timestamp)
            if delta < best_delta:
                best_delta = delta
                best_frame = timestamped_frame
        
        if best_frame is None:
            return None
        
        # Проверяем, что кадр не слишком старый
        frame_age = current_time - best_frame.timestamp
        if frame_age > BUFFER_DURATION_SECONDS:
            return None
    
    # Оцениваем качество кадра (проверяем ВСЕ машины в кадре)
    quality_score, bbox, vehicles_count, all_vehicles = _assess_frame_quality(best_frame.frame, best_frame.timestamp, model)
    
    if quality_score > 0:
        return (quality_score, best_frame, bbox, vehicles_count, all_vehicles)
    return None

def _check_vehicle_movement(vehicles_by_frame: list) -> Dict[int, bool]:
    """
    Проверяет движение машин между кадрами.
    vehicles_by_frame: список [(frame_idx, vehicles_list), ...] отсортированный по времени
    
    Возвращает: {vehicle_idx: is_moving} - движется ли машина
    """
    if len(vehicles_by_frame) < 2:
        return {}
    
    # Трекаем машины между кадрами по позиции
    # Если машина смещается между кадрами - она движется
    MOVEMENT_THRESHOLD = 20.0  # Минимальное смещение в пикселях для определения движения
    
    vehicle_tracks = {}  # track_id -> [(frame_idx, center_x, center_y, size_ratio), ...]
    
    # Простой трекинг: сопоставляем машины между кадрами по близости позиции
    for frame_idx, vehicles in vehicles_by_frame:
        for vehicle in vehicles:
            center_x = vehicle['center_x']
            center_y = vehicle['center_y']
            size_ratio = vehicle['size_ratio']
            
            # Ищем ближайший трек
            best_track = None
            best_distance = float('inf')
            
            for track_id, track_history in vehicle_tracks.items():
                last_frame, last_x, last_y, last_size = track_history[-1]
                if last_frame == frame_idx - 1:  # Предыдущий кадр
                    distance = ((center_x - last_x) ** 2 + (center_y - last_y) ** 2) ** 0.5
                    # Учитываем размер - похожие по размеру машины скорее всего одна и та же
                    size_diff = abs(size_ratio - last_size)
                    if size_diff < 0.02 and distance < 100:  # Похожий размер и близкая позиция
                        if distance < best_distance:
                            best_distance = distance
                            best_track = track_id
            
            if best_track is not None:
                vehicle_tracks[best_track].append((frame_idx, center_x, center_y, size_ratio))
            else:
                # Новый трек
                new_track_id = len(vehicle_tracks)
                vehicle_tracks[new_track_id] = [(frame_idx, center_x, center_y, size_ratio)]
    
    # Определяем какие машины движутся
    moving_vehicles = {}
    for track_id, track_history in vehicle_tracks.items():
        if len(track_history) < 2:
            continue
        
        # Вычисляем общее смещение
        total_movement = 0.0
        for i in range(1, len(track_history)):
            prev_x, prev_y = track_history[i-1][1], track_history[i-1][2]
            curr_x, curr_y = track_history[i][1], track_history[i][2]
            movement = ((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2) ** 0.5
            total_movement += movement
        
        is_moving = total_movement > MOVEMENT_THRESHOLD
        moving_vehicles[track_id] = is_moving
    
    return moving_vehicles

def _compare_vehicles_with_anpr(snow_frame: np.ndarray, anpr_vehicle_image: np.ndarray, model) -> Tuple[float, Optional[dict]]:
    """
    Сравнивает машины на снеговом кадре с машиной с ANPR камеры.
    Использует YOLO для детекции и сравнения размеров/позиций/визуальных признаков.
    Возвращает (match_score, best_vehicle) где match_score 0.0-1.0, best_vehicle - лучшая совпадающая машина.
    """
    if model is None:
        return 0.0, None
    
    try:
        # Детектируем машины на снеговом кадре
        snow_vehicles = _detect_all_vehicles(snow_frame, model)
        
        # Детектируем машину на ANPR фото
        anpr_vehicles = _detect_all_vehicles(anpr_vehicle_image, model)
        
        if not anpr_vehicles or not snow_vehicles:
            return 0.0, None
        
        # Берем самую большую машину с ANPR (скорее всего та что нам нужна)
        anpr_vehicle = max(anpr_vehicles, key=lambda v: v['area'])
        
        fh, fw = snow_frame.shape[:2]
        anh, anw = anpr_vehicle_image.shape[:2]
        
        # Ищем похожую машину на снеговом кадре
        # Критерии: размер (близкий), позиция (ближе к центру = лучше), движение
        best_match_score = 0.0
        best_vehicle = None
        
        for snow_vehicle in snow_vehicles:
            # 1. Сравниваем размер (нормализованный) - важный критерий
            size_diff = abs(snow_vehicle['size_ratio'] - anpr_vehicle['size_ratio'])
            size_score = 1.0 - min(size_diff * 15.0, 1.0)  # Чем ближе размер, тем выше score
            
            # 2. Позиция относительно центра (КРИТИЧНО - машина должна быть по центру)
            center_dist_x = abs((snow_vehicle['center_x'] / fw) - 0.5)  # 0.5 = центр по X
            center_dist_y = abs((snow_vehicle['center_y'] / fh) - 0.55)  # 0.55 = чуть ниже центра по Y
            center_score = 1.0 - (center_dist_x * 2.0 + center_dist_y * 2.0)  # Штраф за отклонение от центра
            center_score = max(0.0, center_score)  # Не может быть отрицательным
            
            # 3. Приоритет большим машинам (ближняя полоса)
            size_bonus = 1.0 if snow_vehicle['size_ratio'] >= 0.08 else 0.5
            
            # 4. Проверка что машина в центральной зоне
            in_zone, _, _, _, _, _, _ = _check_center_zone(snow_vehicle['bbox'], fw, fh)
            zone_bonus = 1.0 if in_zone else 0.3
            
            # Комбинированный score: размер (40%) + позиция (40%) + бонусы (20%)
            match_score = (size_score * 0.4 + center_score * 0.4 + size_bonus * 0.1 + zone_bonus * 0.1)
            
            if match_score > best_match_score:
                best_match_score = match_score
                best_vehicle = snow_vehicle
        
        return best_match_score, best_vehicle
    except Exception as e:
        print(f"[SNOW] Error comparing vehicles with ANPR: {e}")
        import traceback
        print(f"[SNOW] Traceback: {traceback.format_exc()}")
        return 0.0, None

def capture_snow_photo(anpr_vehicle_image_bytes: bytes | None = None) -> Optional[bytes]:
    """
    Захватывает лучший кадр из 5 вариантов (2, 3, 4, 5, 6 секунд назад).
    Если передан anpr_vehicle_image_bytes - использует его для поиска похожей машины.
    Выбирает кадр где машина в центре на ближней полосе (больше по размеру).
    Использует параллельную обработку для ускорения.
    """
    global _frame_buffer, _detection_cache
    import time
    
    print(f"[SNOW] capture_snow_photo: starting capture...")
    if anpr_vehicle_image_bytes:
        print(f"[SNOW] capture_snow_photo: ANPR vehicle image provided for matching")
    current_time = time.time()
    
    # Загружаем ANPR фото если есть
    anpr_image = None
    if anpr_vehicle_image_bytes:
        try:
            import cv2
            import numpy as np
            nparr = np.frombuffer(anpr_vehicle_image_bytes, np.uint8)
            anpr_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if anpr_image is None:
                print(f"[SNOW] Failed to decode ANPR image")
                anpr_vehicle_image_bytes = None
            else:
                print(f"[SNOW] ANPR image loaded: {anpr_image.shape}")
        except Exception as e:
            print(f"[SNOW] Error loading ANPR image: {e}")
            anpr_vehicle_image_bytes = None
            anpr_image = None
    
    # Очищаем старый кэш
    with _detection_cache_lock:
        cache_keys_to_remove = [
            ts for ts in _detection_cache.keys()
            if current_time - ts > DETECTION_CACHE_TTL
        ]
        for key in cache_keys_to_remove:
            del _detection_cache[key]
    
    # Получаем YOLO модель (может быть None если не загрузилась)
    model = _get_yolo_model()
    if model is None:
        print(f"[SNOW] capture_snow_photo: YOLO model not available, using fallback")
    
    # Целевые временные метки для 3 кадров (уменьшено с 5 для экономии памяти)
    # Используем 2, 3, 4 секунды назад (вместо 3, 4, 5, 6, 7)
    target_delays = [2.0, 3.0, 4.0]  # секунд назад
    
    # Параллельная обработка кадров (уменьшено с 3 до 2 потоков для экономии памяти)
    all_frames_data = []  # Все кадры с данными для сравнения: [(delay, quality_score, frame, bbox, vehicles_count, all_vehicles), ...]
    
    if model is not None:
        try:
            # Уменьшено с 3 до 2 потоков для экономии памяти
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(_process_frame_for_delay, delay, current_time, model): delay
                    for delay in target_delays
                }
                
                # Собираем результаты в правильном порядке (по времени)
                results_by_delay = {}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result = future.result(timeout=3.0)  # Таймаут 3 секунды на кадр
                        delay = futures[future]
                        results_by_delay[delay] = result
                    except Exception as e:
                        delay = futures[future]
                        print(f"[SNOW] Error processing frame {delay:.1f}s ago: {e}")
                
                # Обрабатываем кадры в порядке времени (от 2 до 6 секунд)
                for delay in sorted(target_delays):
                    result = results_by_delay.get(delay)
                    if result is not None:
                        quality_score, frame, bbox, vehicles_count, all_vehicles = result
                        all_frames_data.append((delay, quality_score, frame, bbox, vehicles_count, all_vehicles))
                        print(f"[SNOW] Frame {delay:.1f}s ago: quality={quality_score:.1f}, vehicles={vehicles_count}, bbox={bbox}")
                    else:
                        # Кадр обработан, но машины не найдены
                        print(f"[SNOW] Frame {delay:.1f}s ago: no vehicles found")
        except Exception as e:
            print(f"[SNOW] Error in parallel processing: {e}")
    
    # Сравниваем все 5 кадров с ANPR фото (если есть) и выбираем лучший
    if len(all_frames_data) > 0:
        best_frame_data = None
        best_score = -1.0
        best_reason = ""
        
        # Если есть ANPR фото - сравниваем каждый кадр с ним
        if anpr_image is not None and model is not None:
            print(f"[SNOW] Comparing all 5 frames with ANPR image...")
            for delay, quality_score, frame, bbox, vehicles_count, all_vehicles in all_frames_data:
                # Сравниваем машины на этом кадре с машиной на ANPR фото
                match_score, matched_vehicle = _compare_vehicles_with_anpr(
                    frame.frame, anpr_image, model
                )
                
                if match_score > 0 and matched_vehicle:
                    fh, fw = frame.frame.shape[:2]
                    # Проверяем позицию относительно центра
                    center_dist_x = abs((matched_vehicle['center_x'] / fw) - 0.5)
                    center_dist_y = abs((matched_vehicle['center_y'] / fh) - 0.55)
                    center_score = 1.0 - (center_dist_x * 2.0 + center_dist_y * 2.0)
                    center_score = max(0.0, center_score)
                    
                    # Комбинированный score: похожесть с ANPR (60%) + позиция (30%) + размер (10%)
                    combined_score = match_score * 60.0 + center_score * 30.0 + matched_vehicle['size_ratio'] * 100.0
                    
                    print(f"[SNOW] Frame {delay:.1f}s ago: ANPR match={match_score:.3f}, center={center_score:.3f}, size={matched_vehicle['size_ratio']:.3f}, combined={combined_score:.1f}")
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_frame_data = (delay, quality_score, frame, bbox, vehicles_count)
                        best_reason = f"ANPR match ({match_score:.2f}) + center ({center_score:.2f})"
        
        # Если ANPR сравнение не дало результата - используем старую логику (движение + размер)
        if best_frame_data is None:
            print(f"[SNOW] No ANPR match found, using movement-based selection...")
            # Группируем машины по кадрам для анализа движения
            vehicles_by_frame = []
            for delay, quality_score, frame, bbox, vehicles_count, all_vehicles in all_frames_data:
                vehicles_by_frame.append((int(delay), all_vehicles))
            
            # Определяем какие машины движутся
            moving_vehicles_by_frame = {}  # delay -> [vehicle_idx, ...] движущиеся машины
            
            for delay, quality_score, frame, bbox, vehicles_count, all_vehicles in all_frames_data:
                moving_vehicles_by_frame[delay] = []
                
                for vehicle_idx, vehicle in enumerate(all_vehicles):
                    if vehicle['size_ratio'] < 0.08:  # Только большие машины (ближняя полоса)
                        continue
                    
                    # Проверяем есть ли эта машина в других кадрах и смещается ли
                    found_in_other = False
                    total_movement = 0.0
                    last_x, last_y = vehicle['center_x'], vehicle['center_y']
                    
                    for other_delay, _, _, _, _, other_vehicles in all_frames_data:
                        if other_delay == delay:
                            continue
                        
                        # Ищем похожую машину в другом кадре (по размеру и позиции)
                        for other_vehicle in other_vehicles:
                            if other_vehicle['size_ratio'] >= 0.08:  # Тоже большая
                                size_diff = abs(vehicle['size_ratio'] - other_vehicle['size_ratio'])
                                if size_diff < 0.02:  # Похожий размер - возможно та же машина
                                    dist = ((vehicle['center_x'] - other_vehicle['center_x'])**2 + 
                                           (vehicle['center_y'] - other_vehicle['center_y'])**2)**0.5
                                    if dist < 200:  # Близкая позиция или сместилась
                                        found_in_other = True
                                        movement = ((last_x - other_vehicle['center_x'])**2 + 
                                                   (last_y - other_vehicle['center_y'])**2)**0.5
                                        total_movement += movement
                                        last_x, last_y = other_vehicle['center_x'], other_vehicle['center_y']
                                        break
                    
                    # Если машина найдена в других кадрах и сместилась - она движется
                    if found_in_other and total_movement > 30.0:  # Минимум 30 пикселей смещения
                        moving_vehicles_by_frame[delay].append(vehicle_idx)
            
            # Выбираем лучший кадр по движению и размеру
            for delay, quality_score, frame, bbox, vehicles_count, all_vehicles in all_frames_data:
                has_near_lane = False
                has_far_lane = False
                has_moving_near = False
                
                for vehicle_idx, vehicle in enumerate(all_vehicles):
                    if vehicle['size_ratio'] >= 0.08:  # Ближняя полоса
                        has_near_lane = True
                        if vehicle_idx in moving_vehicles_by_frame.get(delay, []):
                            has_moving_near = True
                    elif vehicle['size_ratio'] >= 0.03:  # Дальняя полоса
                        has_far_lane = True
                
                # Оценка: движущаяся ближняя > ближняя > дальняя
                if has_moving_near and quality_score > 0:
                    score = quality_score * 3.0
                    if score > best_score:
                        best_score = score
                        best_frame_data = (delay, quality_score, frame, bbox, vehicles_count)
                        best_reason = f"moving vehicle on near lane"
                elif has_near_lane and quality_score > 0:
                    if quality_score > best_score:
                        best_score = quality_score
                        best_frame_data = (delay, quality_score, frame, bbox, vehicles_count)
                        best_reason = "vehicle on near lane"
                elif has_far_lane and quality_score > 0 and best_frame_data is None:
                    if quality_score > best_score:
                        best_score = quality_score
                        best_frame_data = (delay, quality_score, frame, bbox, vehicles_count)
                        best_reason = "vehicle on far lane (fallback)"
        
        if best_frame_data:
            delay, quality_score, frame_obj, bbox, vehicles_count = best_frame_data
            print(f"[SNOW] capture_snow_photo: selected frame {delay:.1f}s ago with quality={quality_score:.1f}, vehicles={vehicles_count}, reason={best_reason}")
            frame = frame_obj.frame.copy()
            # Освобождаем память после выбора кадра
            del all_frames_data, best_frame_data
        else:
            # Fallback: берем кадр 5 секунд назад без проверки
            print(f"[SNOW] capture_snow_photo: no suitable frames found, using fallback")
            target_timestamp = current_time - 5.0
            with _frame_buffer_lock:
                if len(_frame_buffer) == 0:
                    print(f"[SNOW] capture_snow_photo: buffer is empty, cannot capture")
                    return None
                
                best_frame = None
                best_delta = float('inf')
                for timestamped_frame in _frame_buffer:
                    delta = abs(timestamped_frame.timestamp - target_timestamp)
                    if delta < best_delta:
                        best_delta = delta
                        best_frame = timestamped_frame
                
                if best_frame is None:
                    print(f"[SNOW] capture_snow_photo: no frame found for fallback")
                    return None
                
                frame_age = current_time - best_frame.timestamp
                print(f"[SNOW] capture_snow_photo: fallback frame from {frame_age:.2f}s ago (delta={best_delta:.3f}s)")
                frame = best_frame.frame.copy()
    else:
        # Fallback: берем кадр 5 секунд назад без проверки
        print(f"[SNOW] capture_snow_photo: no frames processed, using fallback")
        target_timestamp = current_time - 5.0
        with _frame_buffer_lock:
            if len(_frame_buffer) == 0:
                print(f"[SNOW] capture_snow_photo: buffer is empty, cannot capture")
                return None
            
            best_frame = None
            best_delta = float('inf')
            for timestamped_frame in _frame_buffer:
                delta = abs(timestamped_frame.timestamp - target_timestamp)
                if delta < best_delta:
                    best_delta = delta
                    best_frame = timestamped_frame
            
            if best_frame is None:
                print(f"[SNOW] capture_snow_photo: no frame found for fallback")
                return None
            
            frame_age = current_time - best_frame.timestamp
            print(f"[SNOW] capture_snow_photo: fallback frame from {frame_age:.2f}s ago (delta={best_delta:.3f}s)")
        frame = best_frame.frame.copy()
    
    # Освобождаем память перед кодированием
    import gc
    if 'anpr_image' in locals() and anpr_image is not None:
        del anpr_image
    if 'all_frames_data' in locals():
        del all_frames_data
    gc.collect()
    
    # Кодируем кадр в JPEG
    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            print(f"[SNOW] capture_snow_photo: failed to encode frame to JPEG")
            return None
        photo_bytes = buf.tobytes()
        print(f"[SNOW] capture_snow_photo: encoded frame, size={len(photo_bytes)} bytes")
        return photo_bytes
    except Exception as e:
        print(f"[SNOW] capture_snow_photo: error encoding frame: {e}")
        import traceback
        print(f"[SNOW] Traceback: {traceback.format_exc()}")
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
        # ВАЖНО: используем frame.copy() чтобы избежать проблем с переиспользованием буфера OpenCV
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
            
            # Логирование размера буфера отключено для уменьшения мусора в логах
            # Раскомментируйте для диагностики:
            # if current_timestamp - last_log_time >= LOG_INTERVAL_SECONDS:
            #     if len(_frame_buffer) > 0:
            #         print(f"[SNOW] buffer size: {len(_frame_buffer)} frames, "
            #               f"oldest: {current_timestamp - _frame_buffer[0].timestamp:.2f}s ago, "
            #               f"newest: {current_timestamp - _frame_buffer[-1].timestamp:.2f}s ago")
            #     last_log_time = current_timestamp

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
