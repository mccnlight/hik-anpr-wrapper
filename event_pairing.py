"""
Очереди снега и номеров + сопоставление по порядку (FIFO).
Снеговой поток при пересечении линии кладёт кадр в snow_queue,
номерная камера при событии кладёт задачу в plate_queue.
Когда в обеих очередях есть элементы — образуется пара (первый снег + первое номерное событие),
она передаётся в общую очередь обработки; воркеры работают параллельно.
"""
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# Максимальная длина очередей (защита от переполнения памяти)
SNOW_QUEUE_MAXLEN = int(os.getenv("PAIRING_SNOW_QUEUE_MAXLEN", "30"))
PLATE_QUEUE_MAXLEN = int(os.getenv("PAIRING_PLATE_QUEUE_MAXLEN", "30"))

# Сопоставление по времени: пара допустима, если разница не больше N секунд (0 = только FIFO)
PAIRING_TIME_WINDOW_SECONDS = float(os.getenv("PAIRING_TIME_WINDOW_SECONDS", "0"))

_lock = threading.Lock()


@dataclass
class SnowItem:
    """Элемент очереди снега: кадр + время."""
    photo_bytes: bytes
    timestamp: float = field(default_factory=time.time)


@dataclass
class PlateItem:
    """Элемент очереди номеров: все данные для фоновой обработки одного события."""
    event_data: Dict[str, Any]
    detection_bytes: bytes | None
    feature_bytes: bytes | None
    license_bytes: bytes | None
    plate_photo_1: bytes | None
    plate_photo_2: bytes | None
    camera_plate_for_gemini: str | None
    main_plate: str | None
    merger: Any
    timestamp: float = field(default_factory=time.time)


# Очереди (FIFO). Доступ только под _lock.
_snow_queue: deque[SnowItem] = deque(maxlen=SNOW_QUEUE_MAXLEN)
_plate_queue: deque[PlateItem] = deque(maxlen=PLATE_QUEUE_MAXLEN)


def push_snow(photo_bytes: bytes) -> None:
    """Добавить кадр снега (вызывается из снегового воркера при пересечении линии)."""
    with _lock:
        if len(_snow_queue) >= _snow_queue.maxlen:
            _snow_queue.popleft()
        _snow_queue.append(SnowItem(photo_bytes=photo_bytes, timestamp=time.time()))


def push_plate(
    event_data: Dict[str, Any],
    detection_bytes: bytes | None,
    feature_bytes: bytes | None,
    license_bytes: bytes | None,
    plate_photo_1: bytes | None,
    plate_photo_2: bytes | None,
    camera_plate_for_gemini: str | None,
    main_plate: str | None,
    merger: Any,
) -> None:
    """Добавить событие номеров (вызывается из API при получении события с камеры)."""
    with _lock:
        if len(_plate_queue) >= _plate_queue.maxlen:
            _plate_queue.popleft()
        _plate_queue.append(
            PlateItem(
                event_data=event_data,
                detection_bytes=detection_bytes,
                feature_bytes=feature_bytes,
                license_bytes=license_bytes,
                plate_photo_1=plate_photo_1,
                plate_photo_2=plate_photo_2,
                camera_plate_for_gemini=camera_plate_for_gemini,
                main_plate=main_plate,
                merger=merger,
                timestamp=time.time(),
            )
        )


def try_match() -> Optional[Tuple[bytes, PlateItem]]:
    """
    Если в обеих очередях есть хотя бы один элемент, вернуть пару (snow_photo_bytes, plate_task).
    По умолчанию (PAIRING_TIME_WINDOW_SECONDS=0): строгий FIFO — первый снег с первым номерным.
    Если PAIRING_TIME_WINDOW_SECONDS>0: та же пара образуется только если разница по времени в пределах окна.
    """
    with _lock:
        if not _snow_queue or not _plate_queue:
            return None

        snow_item = _snow_queue[0]
        plate_item = _plate_queue[0]
        if PAIRING_TIME_WINDOW_SECONDS > 0:
            delta = abs(snow_item.timestamp - plate_item.timestamp)
            if delta > PAIRING_TIME_WINDOW_SECONDS:
                return None

        _snow_queue.popleft()
        _plate_queue.popleft()
        return (snow_item.photo_bytes, plate_item)


def queue_sizes() -> Tuple[int, int]:
    """Текущие размеры очередей (для логов)."""
    with _lock:
        return (len(_snow_queue), len(_plate_queue))
