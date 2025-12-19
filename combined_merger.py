import json
import os
import io
import threading
import time
import asyncio  # нужен для ожидания снеговых событий, если ANPR пришел раньше
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Optional, Tuple

import httpx
from google import genai
from PIL import Image


# Локальный часовой пояс (по умолчанию Astana/UTC+6)
LOCAL_TZ = timezone(
    timedelta(hours=int(os.getenv("LOCAL_TZ_OFFSET_HOURS", "6")))
)

# Максимальный возраст события для матчинга (секунды)
# События старше этого возраста не будут матчиться с ANPR событиями
# Это предотвращает матчинг старых событий от стоячих машин с новыми ANPR событиями
MAX_EVENT_AGE_SECONDS = float(os.getenv("MERGE_MAX_EVENT_AGE_SECONDS", "30.0"))  # Увеличено до 30 секунд для учета задержек в создании снеговых событий
# Сколько секунд максимум ждать прихода парного события снега, если ANPR пришел раньше.
# По умолчанию равно MERGE_WINDOW_SECONDS (если задан), иначе 20.
WAIT_FOR_SNOW_SECONDS = float(
    os.getenv(
        "MERGE_WAIT_FOR_SNOW_SECONDS",
        os.getenv("MERGE_WINDOW_SECONDS", "20")
    )
)

# Требовать ли обязательный матч со снегом, иначе не отправлять событие
REQUIRE_SNOW_MATCH = os.getenv("MERGE_REQUIRE_SNOW_MATCH", "false").lower() == "true"


def _parse_iso_dt(value: str | None) -> Optional[datetime]:
    """
    Parse ISO8601 datetime string into aware UTC datetime.
    Accepts trailing "Z" and timezone offsets like "+06:00".
    """
    if not value:
        return None
    try:
        cleaned = str(value).strip()
        
        # Проверяем, нет ли дублирования timezone (например, +00:00Z или +00:00+00:00)
        if cleaned.endswith("+00:00Z") or cleaned.endswith("-00:00Z"):
            # Убираем дублирование: оставляем только Z
            cleaned = cleaned[:-6] + "Z"
        elif "+00:00+00:00" in cleaned or "-00:00+00:00" in cleaned:
            # Убираем дублирование timezone
            cleaned = cleaned.replace("+00:00+00:00", "+00:00").replace("-00:00+00:00", "+00:00")
        
        # Если заканчивается на Z, заменяем на +00:00
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        # Парсим ISO формат (поддерживает +06:00, -05:00 и т.д.)
        dt = datetime.fromisoformat(cleaned)
        # Если нет timezone, добавляем локальный TZ
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        # Конвертируем в локальный TZ
        return dt.astimezone(LOCAL_TZ)
    except Exception as e:
        print(f"[MERGER] ERROR parsing datetime '{value}': {e}")
        return None


def _now() -> datetime:
    """Возвращает текущее время в локальном часовом поясе (по умолчанию UTC+6 Astana)."""
    return datetime.now(tz=LOCAL_TZ)


@dataclass
class SnowEvent:
    event_time: datetime
    payload: Dict[str, Any]
    photo_bytes: bytes | None


@dataclass
class ANPREvent:
    event_time: datetime
    event_data: Dict[str, Any]
    detection_bytes: bytes | None
    feature_bytes: bytes | None
    license_bytes: bytes | None


class EventMerger:
    """
    Keeps snow events and ANPR events in memory and merges them
    when either arrives (supports both snow->ANPR and ANPR->snow order).
    """

    def __init__(
        self,
        upstream_url: str,
        window_seconds: int = 30,
        ttl_seconds: int = 60,
    ):
        self.upstream_url = upstream_url
        self.window = timedelta(seconds=window_seconds)
        self.ttl = timedelta(seconds=ttl_seconds)
        self._snow_events: Deque[SnowEvent] = deque()
        self._anpr_events: Deque[ANPREvent] = deque()  # Очередь для ANPR событий, которые пришли раньше снега
        self._lock = threading.Lock()
        self._gemini_client: genai.Client | None = None
        self._gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self._gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        # Отметки обработанных ANPR, чтобы не запускать Gemini повторно при дубликатах
        # key: (plate, event_time_iso) -> stored_time
        self._processed_anpr: Dict[Tuple[str, str], datetime] = {}
        # Опциональная проверка whitelist перед Gemini
        self._vehicle_check_url = os.getenv(
            "MERGER_VEHICLE_CHECK_URL",
            "https://snowops-anpr-service.onrender.com/internal/vehicles/check",
        )
        self._vehicle_check_token = os.getenv("MERGER_VEHICLE_CHECK_TOKEN", "")
        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True, name="merger-cleanup"
        )
        self._cleanup_thread.start()

    def _cleanup(self, now: datetime) -> None:
        # Очистка снеговых событий - удаляем события старше TTL
        # Также удаляем события, которые слишком старые для матчинга (старше MAX_EVENT_AGE_SECONDS + небольшой запас)
        removed_snow = 0
        while self._snow_events:
            oldest = self._snow_events[0]
            age = (now - oldest.event_time).total_seconds()
            # Удаляем если старше TTL или слишком старые для матчинга
            if age > self.ttl.total_seconds() or age > (MAX_EVENT_AGE_SECONDS + 5.0):
                self._snow_events.popleft()
                removed_snow += 1
            else:
                break
        if removed_snow > 0:
            print(f"[MERGER] cleaned up {removed_snow} old snow events (age > {MAX_EVENT_AGE_SECONDS + 5.0}s or TTL)")
        
        # Очистка ANPR событий - удаляем события старше TTL
        removed_anpr = 0
        while self._anpr_events:
            oldest = self._anpr_events[0]
            age = (now - oldest.event_time).total_seconds()
            if age > self.ttl.total_seconds():
                self._anpr_events.popleft()
                removed_anpr += 1
            else:
                break
        if removed_anpr > 0:
            print(f"[MERGER] cleaned up {removed_anpr} old ANPR events (age > TTL)")

        # Очистка processed меток
        removed_processed = 0
        ttl_seconds = self.ttl.total_seconds()
        for key, ts in list(self._processed_anpr.items()):
            if (now - ts).total_seconds() > ttl_seconds:
                del self._processed_anpr[key]
                removed_processed += 1
        if removed_processed > 0:
            print(f"[MERGER] cleaned up {removed_processed} processed ANPR marks (age > TTL)")

    def add_snow_event(self, payload: Dict[str, Any], photo_bytes: bytes | None) -> None:
        event_time_str = str(payload.get("event_time", ""))
        event_time = _parse_iso_dt(event_time_str) or _now()
        now = _now()
        
        # Логируем разницу между временем события и текущим временем для диагностики
        time_diff = (now - event_time).total_seconds()
        print(
            f"[MERGER] storing snow event: event_time={event_time.isoformat()}, "
            f"now={now.isoformat()}, time_diff={time_diff:.2f}s, "
            f"original_str='{event_time_str}', photo_size={len(photo_bytes) if photo_bytes else 0} bytes"
        )
        
        snow_payload = dict(payload)
        snow_payload["event_time"] = event_time.isoformat()
        snow_event_obj = SnowEvent(event_time, snow_payload, photo_bytes)
        
        with self._lock:
            self._cleanup(now)
            
            # Пытаемся найти совпадение с ANPR событием, которое пришло раньше
            anpr_match = self._pop_anpr_match(event_time)
            if anpr_match:
                print(f"[MERGER] found ANPR match for snow event (ANPR came first), will process in background thread")
                # Запускаем обработку в отдельном потоке
                def process_in_thread():
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                    loop.run_until_complete(self._process_matched_pair(anpr_match, snow_event_obj))
                
                thread = threading.Thread(target=process_in_thread, daemon=True)
                thread.start()
                return
            
            # Если ANPR не найден, просто добавляем снег в очередь
            self._snow_events.append(snow_event_obj)
        
        print(
            f"[MERGER] stored snow event at {event_time.isoformat()}, "
            f"queue size={len(self._snow_events)}"
        )

    def _is_processed(self, plate: str, event_time_iso: str) -> bool:
        return (plate, event_time_iso) in self._processed_anpr

    def _mark_processed(self, plate: str, event_time_iso: str, ts: datetime) -> None:
        self._processed_anpr[(plate, event_time_iso)] = ts

    def _pop_anpr_match(self, snow_time: datetime) -> Optional[ANPREvent]:
        """
        Ищет ANPR событие, которое соответствует снеговому событию по времени.
        Используется когда снег приходит раньше ANPR.
        """
        best_idx = None
        best_delta = None
        
        now = _now()
        
        print(f"[MERGER] DEBUG: searching ANPR match for snow_time={snow_time.isoformat()}, anpr_queue_size={len(self._anpr_events)}, window={self.window.total_seconds()}s")
        
        if len(self._anpr_events) == 0:
            print(f"[MERGER] DEBUG: ANPR queue is empty, no match possible")
            return None
        
        for idx, anpr_event in enumerate(self._anpr_events):
            delta = snow_time - anpr_event.event_time
            delta_abs = abs(delta)
            delta_abs_sec = delta_abs.total_seconds()
            age_from_now = (now - anpr_event.event_time).total_seconds()
            
            print(f"[MERGER] DEBUG: anpr[{idx}] time={anpr_event.event_time.isoformat()}, "
                  f"age_from_now={age_from_now:.1f}s, delta={delta.total_seconds():.1f}s, |delta|={delta_abs_sec:.1f}s")
            
            # Пропускаем слишком старые ANPR события (даже если они в окне) - это могут быть события от уезжающих машин
            if age_from_now > MAX_EVENT_AGE_SECONDS:
                print(f"[MERGER] DEBUG: anpr[{idx}] age_from_now={age_from_now:.1f}s > MAX_EVENT_AGE_SECONDS={MAX_EVENT_AGE_SECONDS}s, skipping (likely from exiting vehicle)")
                continue
            
            if delta_abs <= self.window:
                if delta_abs_sec > MAX_EVENT_AGE_SECONDS:
                    print(f"[MERGER] DEBUG: anpr[{idx}] |delta|={delta_abs_sec:.1f}s > MAX_EVENT_AGE_SECONDS={MAX_EVENT_AGE_SECONDS}s, skipping")
                    continue
                if best_delta is None or delta_abs < best_delta:
                    best_delta = delta_abs
                    best_idx = idx
                    print(f"[MERGER] DEBUG: anpr[{idx}] is candidate, |delta|={delta_abs_sec:.1f}s")
            else:
                print(f"[MERGER] DEBUG: anpr[{idx}] |delta|={delta_abs_sec:.1f}s exceeds window={self.window.total_seconds()}s")
        
        if best_idx is None:
            print(f"[MERGER] DEBUG: no ANPR match found after checking {len(self._anpr_events)} events")
            # Дополнительная диагностика: показываем ближайшее событие
            if len(self._anpr_events) > 0:
                closest_idx = 0
                closest_delta = abs((snow_time - self._anpr_events[0].event_time).total_seconds())
                for idx, anpr_event in enumerate(self._anpr_events):
                    delta_abs_sec = abs((snow_time - anpr_event.event_time).total_seconds())
                    if delta_abs_sec < closest_delta:
                        closest_delta = delta_abs_sec
                        closest_idx = idx
                closest_event = self._anpr_events[closest_idx]
                print(f"[MERGER] DEBUG: closest ANPR event is anpr[{closest_idx}] with |delta|={closest_delta:.1f}s "
                      f"(window={self.window.total_seconds()}s, max_age={MAX_EVENT_AGE_SECONDS}s)")
            return None
        
        match = self._anpr_events[best_idx]
        del self._anpr_events[best_idx]
        print(f"[MERGER] matched ANPR[{best_idx}] with snow event, delta={best_delta.total_seconds():.1f}s")
        return match

    def _pop_match(self, anpr_time: datetime) -> Optional[SnowEvent]:
        best_idx = None
        best_delta = None

        print(f"[MERGER] DEBUG: searching match for anpr_time={anpr_time.isoformat()}, queue_size={len(self._snow_events)}, window={self.window.total_seconds()}s")
        
        if len(self._snow_events) == 0:
            print(f"[MERGER] DEBUG: queue is empty, no match possible")
            return None
        
        # Логируем все события в очереди для диагностики
        now = _now()
        anpr_time_diff_from_now = (anpr_time - now).total_seconds()
        print(f"[MERGER] DEBUG: current time={now.isoformat()}, anpr_time={anpr_time.isoformat()}, "
              f"anpr_time_diff_from_now={anpr_time_diff_from_now:.1f}s")
        
        # Если ANPR время сильно отличается от реального (больше чем окно), используем текущее время для матчинга
        # Это защита от неправильно настроенных часов на камере
        use_current_time_for_matching = abs(anpr_time_diff_from_now) > self.window.total_seconds()
        match_time = now if use_current_time_for_matching else anpr_time
        
        if use_current_time_for_matching:
            print(f"[MERGER] WARNING: ANPR time differs from current time by {abs(anpr_time_diff_from_now):.1f}s "
                  f"(>{self.window.total_seconds()}s), using current time for matching instead of ANPR time")
        
        for idx, snow_event in enumerate(self._snow_events):
            age_from_now = (now - snow_event.event_time).total_seconds()
            delta = match_time - snow_event.event_time
            delta_seconds = delta.total_seconds()
            delta_abs = abs(delta)
            delta_abs_sec = delta_abs.total_seconds()
            print(f"[MERGER] DEBUG: snow[{idx}] time={snow_event.event_time.isoformat()}, "
                  f"age_from_now={age_from_now:.1f}s, delta={delta_seconds:.1f}s, |delta|={delta_abs_sec:.1f}s, "
                  f"has_photo={snow_event.photo_bytes is not None}, match_time={match_time.isoformat()}")
            
            # Пропускаем слишком старые события (даже если они в окне) - это могут быть события от уезжающих машин
            if age_from_now > MAX_EVENT_AGE_SECONDS:
                print(f"[MERGER] DEBUG: snow[{idx}] age_from_now={age_from_now:.1f}s > MAX_EVENT_AGE_SECONDS={MAX_EVENT_AGE_SECONDS}s, skipping (likely from exiting vehicle)")
                continue
            
            # Матчим по модулю, чтобы не терять случаи, когда ANPR-время чуть раньше
            if delta_abs <= self.window:
                # Дополнительная проверка: если разница по модулю больше MAX_EVENT_AGE_SECONDS, пропускаем
                # (даже если событие в окне, оно может быть от другой машины)
                if delta_abs_sec > MAX_EVENT_AGE_SECONDS:
                    print(f"[MERGER] DEBUG: snow[{idx}] |delta|={delta_abs_sec:.1f}s > MAX_EVENT_AGE_SECONDS={MAX_EVENT_AGE_SECONDS}s, skipping")
                    continue
                if best_delta is None or delta_abs < best_delta:
                    best_delta = delta_abs
                    best_idx = idx
                    print(f"[MERGER] DEBUG: snow[{idx}] is candidate, |delta|={delta_abs_sec:.1f}s (raw delta={delta_seconds:.1f}s)")
            else:
                print(f"[MERGER] DEBUG: snow[{idx}] |delta|={delta_abs_sec:.1f}s exceeds window={self.window.total_seconds()}s")

        if best_idx is None:
            print(f"[MERGER] DEBUG: no match found after checking {len(self._snow_events)} events")
            # Дополнительная диагностика: показываем ближайшее событие
            if len(self._snow_events) > 0:
                closest_idx = 0
                closest_delta = abs((match_time - self._snow_events[0].event_time).total_seconds())
                for idx, snow_event in enumerate(self._snow_events):
                    delta_abs_sec = abs((match_time - snow_event.event_time).total_seconds())
                    if delta_abs_sec < closest_delta:
                        closest_delta = delta_abs_sec
                        closest_idx = idx
                closest_event = self._snow_events[closest_idx]
                print(f"[MERGER] DEBUG: closest event is snow[{closest_idx}] with |delta|={closest_delta:.1f}s "
                      f"(window={self.window.total_seconds()}s, max_age={MAX_EVENT_AGE_SECONDS}s, match_time={match_time.isoformat()})")
            return None

        # remove matched event
        match = self._snow_events[best_idx]
        del self._snow_events[best_idx]
        print(f"[MERGER] DEBUG: matched snow[{best_idx}], delta={best_delta.total_seconds():.1f}s, "
              f"photo_size={len(match.photo_bytes) if match.photo_bytes else 0} bytes")
        return match

    def restore_snow_event(self, snow_event: SnowEvent) -> None:
        """
        Возвращает снеговое событие обратно в очередь.
        Используется когда событие номеров не было сохранено (машины нет в базе).
        """
        with self._lock:
            # Вставляем событие обратно в очередь, сохраняя порядок по времени
            inserted = False
            for idx, existing_event in enumerate(self._snow_events):
                if snow_event.event_time <= existing_event.event_time:
                    self._snow_events.insert(idx, snow_event)
                    inserted = True
                    break
            if not inserted:
                # Если событие самое новое, добавляем в конец
                self._snow_events.append(snow_event)
        print(
            f"[MERGER] restored snow event at {snow_event.event_time.isoformat()}, "
            f"queue size={len(self._snow_events)}"
        )

    def _cleanup_loop(self) -> None:
        """
        Периодически чистит просроченные снеговые события и логирует состояние очереди.
        Проверка происходит каждые 2 секунды для диагностики.
        """
        interval = 2.0  # Проверка каждые 2 секунды
        while not self._stop_cleanup.is_set():
            time.sleep(interval)
            now = _now()
            with self._lock:
                self._cleanup(now)
                # Логируем состояние очередей для диагностики
                if len(self._snow_events) > 0:
                    oldest_age = (now - self._snow_events[0].event_time).total_seconds()
                    newest_age = (now - self._snow_events[-1].event_time).total_seconds()
                    print(f"[MERGER] SNOW QUEUE STATUS: size={len(self._snow_events)}, "
                          f"oldest_age={oldest_age:.1f}s, newest_age={newest_age:.1f}s, "
                          f"window={self.window.total_seconds()}s, ttl={self.ttl.total_seconds()}s")
                if len(self._anpr_events) > 0:
                    oldest_age = (now - self._anpr_events[0].event_time).total_seconds()
                    newest_age = (now - self._anpr_events[-1].event_time).total_seconds()
                    print(f"[MERGER] ANPR QUEUE STATUS: size={len(self._anpr_events)}, "
                          f"oldest_age={oldest_age:.1f}s, newest_age={newest_age:.1f}s, "
                          f"window={self.window.total_seconds()}s, ttl={self.ttl.total_seconds()}s")

    def _get_gemini_client(self) -> genai.Client:
        if self._gemini_client is None:
            if not self._gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._gemini_client = genai.Client(api_key=self._gemini_api_key)
        return self._gemini_client

    async def _check_vehicle_exists(self, plate: str) -> Optional[bool]:
        """
        Быстрая проверка whitelist по HTTP, чтобы не дергать Gemini для несуществующих машин.
        Ожидается 2xx, если машина найдена; 404/403 — нет; 5xx — считаем недоступно (None).
        """
        if not self._vehicle_check_url or not plate:
            return None
        try:
            headers = {}
            if self._vehicle_check_token:
                headers["Authorization"] = f"Bearer {self._vehicle_check_token}"
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self._vehicle_check_url, params={"plate": plate}, headers=headers)
                if resp.status_code >= 500:
                    return None
                return resp.status_code < 300
        except Exception as e:
            print(f"[MERGER] vehicle check failed for plate={plate}: {e}")
            return None

    async def analyze_with_gemini(
        self,
        snow_photo: bytes,
        plate_photo_1: bytes,
        plate_photo_2: bytes | None,
        camera_plate: str | None = None,
    ) -> Dict[str, Any]:
        """
        Анализирует 3 фотографии через Gemini:
        - snow_photo: фото снега (обязательно)
        - plate_photo_1: первое фото номера (обязательно, обычно detectionPicture - обычная фотка)
        - plate_photo_2: второе фото номера (опционально, featurePicture или licensePlatePicture - приближенная фотка)
        - camera_plate: номер от камеры (может быть неверным, нужна дополнительная проверка)
        
        Возвращает:
        {
            "snow_percentage": 0.0-100.0,
            "snow_confidence": 0.0-1.0,
            "plate": "номер или None",
            "plate_confidence": 0.0-1.0,
            "error": "описание ошибки если есть"
        }
        """
        if not self._gemini_api_key:
            print("[GEMINI] ERROR: GEMINI_API_KEY is not set")
            return {
                "error": "GEMINI_API_KEY is not set",
                "snow_percentage": 0.0,
                "snow_confidence": 0.0,
                "plate": None,
                "plate_confidence": 0.0,
            }

        try:
            import time as time_module
            start_time = time_module.time()
            
            # Загружаем изображения
            snow_image = Image.open(io.BytesIO(snow_photo)).convert("RGB")
            plate_image_1 = Image.open(io.BytesIO(plate_photo_1)).convert("RGB")
            
            images = [snow_image, plate_image_1]
            if plate_photo_2:
                plate_image_2 = Image.open(io.BytesIO(plate_photo_2)).convert("RGB")
                images.append(plate_image_2)
            
            print(f"[GEMINI] Starting analysis: snow_image={snow_image.size}, "
                  f"plate_image_1={plate_image_1.size}, "
                  f"plate_image_2={'present' if plate_photo_2 else 'none'}")

            image3_text = "IMAGE 3: License plate photo 2 - close-up/zoomed view of the license plate (approximated photo).\n" if plate_photo_2 else ""
            camera_plate_text = ""
            if camera_plate:
                camera_plate_text = (
                    f"\nIMPORTANT: The camera detected plate number '{camera_plate}', but this may be INCORRECT. "
                    "You must verify and correct it by carefully reading the actual plate from the images. "
                    "Use the camera's suggestion only as a hint, but always verify against what you see in the photos.\n"
                )
            
            prompt = (
                "You are analyzing truck photos for snow volume and license plate recognition.\n\n"
                "IMAGE 1: Snow photo - shows the cargo bed of a truck.\n"
                "IMAGE 2: License plate photo 1 - shows the vehicle's license plate (normal/wide view).\n"
                + image3_text +
                camera_plate_text +
                "\n"
                "CRITICAL: Focus ONLY on the truck that is CLOSEST to the camera (on the nearest lane). "
                "If there are multiple trucks in the images, analyze ONLY the one that appears largest/closest. "
                "Ignore any trucks that are further away or in the background.\n\n"
                "TASKS:\n"
                "1. Analyze IMAGE 1 (snow photo):\n"
                "   - Identify the truck that is CLOSEST to the camera (largest in the frame, on the nearest lane).\n"
                "   - Analyze ONLY the cargo bed of THIS nearest truck.\n"
                "   - Classify ONLY loose/bulk snow inside the OPEN cargo bed of the nearest truck.\n"
                "   - Exclude: painted/clean metal or plastic surfaces, tarps, roof/hood, sides of the truck,\n"
                "     sun glare, white paint, reflections, frost/ice, road, background, or closed/covered beds.\n"
                "   - If the bed of the nearest truck is not clearly visible or is closed/covered/fully outside the frame, set snow_percentage=0 and snow_confidence=0.0.\n"
                "   - Snow must look like uneven/loose material with texture; a smooth flat surface (even if white) is NOT snow.\n"
                "   - DO NOT analyze snow in trucks that are further away or in the background.\n"
                "\n"
                "2. Recognize license plate from IMAGE 2 (and IMAGE 3 if provided):\n"
                "   - Identify the truck that is CLOSEST to the camera (largest in the frame, on the nearest lane).\n"
                "   - Extract the license plate number from THIS nearest truck ONLY.\n"
                "   - You have TWO photos: IMAGE 2 is normal/wide view, IMAGE 3 (if provided) is close-up/zoomed view.\n"
                "   - Use BOTH photos to get the most accurate result - the close-up (IMAGE 3) usually has better detail.\n"
                "   - Kazakhstan license plate format is STRICT:\n"
                "     * Format 1: 111AAA11 (3 digits, 3 letters, 2 digits) - example: 035AL115\n"
                "     * Format 2: 111AA11 (3 digits, 2 letters, 2 digits) - example: 035AL15\n"
                "     * Region codes: 01-18 ONLY (there is NO region 19)\n"
                "     * Letters are Latin (A-Z), NOT Cyrillic\n"
                "   - The plate number MUST match one of these formats exactly.\n"
                "   - If you cannot clearly read a valid format from the nearest truck, return null for plate.\n"
                "   - Return the plate number WITHOUT spaces, dashes, or other separators (e.g., '035AL115' not '035 AL 115').\n"
                "   - DO NOT read plates from trucks that are further away or in the background.\n"
                "\n"
                "Return JSON with fields:\n"
                "- snow_percentage: 0.0-100.0 (how full the bed is with snow, 0-100 scale)\n"
                "- snow_confidence: 0.0-1.0 (confidence in snow analysis)\n"
                "- plate: string or null (recognized license plate number in format 111AAA11 or 111AA11, or null if not recognized)\n"
                "- plate_confidence: 0.0-1.0 (confidence in plate recognition)\n\n"
                "Example:\n"
                "{\n"
                '  "snow_percentage": 42.5,\n'
                '  "snow_confidence": 0.9,\n'
                '  "plate": "035AL115",\n'
                '  "plate_confidence": 0.85\n'
                "}\n"
            )
            
            print(f"[GEMINI] Sending request to model={self._gemini_model}, "
                  f"images_count={len(images)}, prompt_length={len(prompt)} chars")

            client = self._get_gemini_client()
            response = client.models.generate_content(
                model=self._gemini_model,
                contents=images + [prompt],
            )
            
            request_duration = time_module.time() - start_time
            text = (response.text or "").strip()
            print(f"[GEMINI] Response received in {request_duration:.2f}s, response_length={len(text)} chars")
            print(f"[GEMINI] Raw response (first 500 chars): {text[:500]}")
            
            if not text:
                print("[GEMINI] ERROR: Empty response from Gemini")
                return {
                    "error": "Empty response from Gemini",
                    "snow_percentage": 0.0,
                    "snow_confidence": 0.0,
                    "plate": None,
                    "plate_confidence": 0.0,
                }

            # Очищаем ответ от markdown
            original_text = text
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
                print(f"[GEMINI] Cleaned response (removed markdown): length={len(text)} chars")

            try:
                result = json.loads(text)
                print(f"[GEMINI] Successfully parsed JSON: {result}")
                
                # Нормализуем результат
                snow_percentage = result.get("snow_percentage", 0.0)
                snow_confidence = result.get("snow_confidence", 0.0)
                plate = result.get("plate")
                plate_confidence = result.get("plate_confidence", 0.0)
                
                # Нормализуем snow_percentage (может быть 0-1 или 0-100)
                try:
                    snow_percentage = float(snow_percentage)
                    if 0.0 <= snow_percentage <= 1.0:
                        snow_percentage = snow_percentage * 100.0
                    snow_percentage = max(0.0, min(100.0, round(snow_percentage, 2)))
                except (ValueError, TypeError):
                    snow_percentage = 0.0
                
                # Нормализуем confidence значения
                try:
                    snow_confidence = float(snow_confidence)
                    snow_confidence = max(0.0, min(1.0, snow_confidence))
                except (ValueError, TypeError):
                    snow_confidence = 0.0
                
                try:
                    plate_confidence = float(plate_confidence)
                    plate_confidence = max(0.0, min(1.0, plate_confidence))
                except (ValueError, TypeError):
                    plate_confidence = 0.0
                
                # Нормализуем номер (убираем пробелы, приводим к верхнему регистру)
                if plate:
                    plate = str(plate).strip().upper().replace(" ", "")
                    if not plate or plate == "NULL" or plate == "NONE":
                        plate = None
                
                return {
                    "snow_percentage": snow_percentage,
                    "snow_confidence": snow_confidence,
                    "plate": plate,
                    "plate_confidence": plate_confidence,
                }
            except json.JSONDecodeError as e:
                print(f"[GEMINI] ERROR: JSON parse failed: {e}")
                print(f"[GEMINI] Failed to parse text: {text[:500]}")
                return {
                    "raw": original_text,
                    "error": f"JSON parse error: {e}",
                    "snow_percentage": 0.0,
                    "snow_confidence": 0.0,
                    "plate": None,
                    "plate_confidence": 0.0,
                }
        except Exception as e:
            print(f"[GEMINI] EXCEPTION: {type(e).__name__}: {e}")
            import traceback
            print(f"[GEMINI] Traceback: {traceback.format_exc()}")
            return {
                "error": str(e),
                "snow_percentage": 0.0,
                "snow_confidence": 0.0,
                "plate": None,
                "plate_confidence": 0.0,
            }

    def _analyze_snow_gemini(
        self, photo_bytes: bytes, bbox: Optional[Any]
    ) -> Dict[str, Any]:
        """
        Analyze truck bed fill via Gemini using in-memory JPEG bytes.
        (Старая функция, оставлена для обратной совместимости, но больше не используется)
        """
        if not self._gemini_api_key:
            print("[GEMINI] ERROR: GEMINI_API_KEY is not set")
            return {"error": "GEMINI_API_KEY is not set"}

        try:
            import time as time_module
            start_time = time_module.time()
            
            image = Image.open(io.BytesIO(photo_bytes)).convert("RGB")
            original_size = (image.width, image.height)
            print(f"[GEMINI] Starting analysis: original image size={original_size}, bbox={bbox}")

            if bbox:
                try:
                    x1, y1, x2, y2 = map(int, bbox)
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(image.width, x2 + padding)
                    y2 = min(image.height, y2 + padding)
                    image = image.crop((x1, y1, x2, y2))
                    print(f"[GEMINI] Cropped image: size=({image.width}, {image.height}), crop=({x1},{y1},{x2},{y2})")
                except Exception as e:
                    print(f"[GEMINI] WARNING: Failed to crop image: {e}")

            prompt = (
                "You see the OPEN cargo bed of a truck. Classify ONLY loose/bulk snow inside the bed.\n"
                "Critical exclusions: painted/clean metal or plastic surfaces, tarps, roof/hood, sides of the truck,\n"
                "sun glare, white paint, reflections, frost/ice, road, background, or closed/covered beds.\n"
                "If the bed is not clearly visible or is closed/covered/fully outside the frame, set percentage=0 and confidence=0.0.\n"
                "Snow must look like uneven/loose material with texture; a smooth flat surface (even if white) is NOT snow.\n"
                "Return JSON with fields:\n"
                "- percentage: 0.0-1.0 or 0-100 for how full with snow\n"
                "- confidence: 0.0-1.0\n\n"
                "Example:\n"
                "{\n"
                '  \"percentage\": 0.42,\n'
                '  \"confidence\": 0.9\n'
                "}\n"
            )
            print(f"[GEMINI] Sending request to model={self._gemini_model}, prompt_length={len(prompt)} chars")

            client = self._get_gemini_client()
            response = client.models.generate_content(
                model=self._gemini_model,
                contents=[image, prompt],
            )
            
            request_duration = time_module.time() - start_time
            text = (response.text or "").strip()
            print(f"[GEMINI] Response received in {request_duration:.2f}s, response_length={len(text)} chars")
            print(f"[GEMINI] Raw response (first 200 chars): {text[:200]}")
            
            if not text:
                print("[GEMINI] ERROR: Empty response from Gemini")
                return {"error": "Empty response from Gemini"}

            original_text = text
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
                print(f"[GEMINI] Cleaned response (removed markdown): length={len(text)} chars")

            try:
                result = json.loads(text)
                print(f"[GEMINI] Successfully parsed JSON: {result}")
                return result
            except json.JSONDecodeError as e:
                print(f"[GEMINI] ERROR: JSON parse failed: {e}")
                print(f"[GEMINI] Failed to parse text: {text[:500]}")
                return {"raw": original_text, "error": f"JSON parse error: {e}", "percentage": 0.0, "confidence": 0.0}
        except Exception as e:
            print(f"[GEMINI] EXCEPTION: {type(e).__name__}: {e}")
            import traceback
            print(f"[GEMINI] Traceback: {traceback.format_exc()}")
            return {"error": str(e), "percentage": 0.0, "confidence": 0.0}

    @staticmethod
    def _extract_snow_fields(gemini_result: dict) -> tuple[Optional[float], Optional[float]]:
        percentage = None
        confidence = None
        raw = None

        if isinstance(gemini_result, dict):
            percentage = gemini_result.get("percentage")
            confidence = gemini_result.get("confidence")
            raw = gemini_result.get("raw")

        if (percentage is None or confidence is None) and raw:
            raw_s = str(raw).strip()
            try:
                if raw_s.startswith("```"):
                    raw_s = raw_s.strip("`")
                    if raw_s.lower().startswith("json"):
                        raw_s = raw_s[4:].strip()
                parsed = json.loads(raw_s)
                percentage = parsed.get("percentage") if percentage is None else percentage
                confidence = parsed.get("confidence") if confidence is None else confidence
            except Exception:
                pass

        try:
            if percentage is not None:
                val = float(percentage)
                if 0.0 <= val <= 1.0:
                    percentage = round(val * 100, 2)
                elif 0 <= val <= 100:
                    percentage = round(val, 2)
                else:
                    percentage = max(0.0, min(100.0, round(val, 2)))
        except Exception:
            percentage = None

        try:
            if confidence is not None:
                confidence = float(confidence)
        except Exception:
            confidence = None

        return percentage, confidence

    async def _process_matched_pair(self, anpr_event_obj: ANPREvent, snow_event: SnowEvent) -> Dict[str, Any]:
        """
        Обрабатывает совпавшую пару ANPR и снег (когда снег пришел раньше ANPR).
        """
        try:
            result = await self._combine_and_send_internal(
                anpr_event_obj.event_data,
                anpr_event_obj.detection_bytes,
                anpr_event_obj.feature_bytes,
                anpr_event_obj.license_bytes,
                snow_event
            )
            print(f"[MERGER] processed matched pair (snow->ANPR): sent={result.get('sent')}, status={result.get('status')}")
            return result
        except Exception as e:
            print(f"[MERGER] ERROR processing matched pair: {e}")
            import traceback
            print(f"[MERGER] Traceback: {traceback.format_exc()}")
            return {"sent": False, "error": str(e)}

    async def combine_and_send(
        self,
        anpr_event: Dict[str, Any],
        detection_bytes: bytes | None,
        feature_bytes: bytes | None,
        license_bytes: bytes | None,
    ) -> Dict[str, Any]:
        """
        Merge ANPR event with the closest earlier snow event (within window)
        and send a single multipart request upstream.
        Поддерживает оба порядка: ANPR->snow и snow->ANPR.
        """
        now = _now()
        anpr_time_str = str(anpr_event.get("event_time", ""))
        anpr_time = _parse_iso_dt(anpr_time_str) or now
        plate = str(anpr_event.get("plate") or "")
        
        # Логируем разницу между временем ANPR и текущим временем для диагностики
        anpr_time_diff = (now - anpr_time).total_seconds()
        print(f"[MERGER] DEBUG: anpr_event_time_str='{anpr_time_str}', parsed={anpr_time.isoformat()}, "
              f"now={now.isoformat()}, anpr_time_diff={anpr_time_diff:.2f}s")

        with self._lock:
            # ВАЖНО: очистка должна использовать текущее время (now), а не anpr_time!
            # Если anpr_time в будущем (неправильные часы камеры), очистка по anpr_time удалит все события
            self._cleanup(now)
            snow_event = self._pop_match(anpr_time)
            
            # Если снег не найден, добавляем ANPR событие в очередь для последующего матчинга
            # (когда снег придет позже, он найдет это ANPR событие)
            if snow_event is None:
                anpr_event_obj = ANPREvent(anpr_time, anpr_event, detection_bytes, feature_bytes, license_bytes)
                self._anpr_events.append(anpr_event_obj)
                print(f"[MERGER] no snow match yet, stored ANPR event in queue (queue_size={len(self._anpr_events)})")
            else:
                # Если снег найден, не добавляем ANPR в очередь, так как он будет обработан сразу
                print(f"[MERGER] found snow match immediately (snow came first)")

        # Если снег еще не пришел, и разрешено подождать — ждём до заданного таймаута
        # Проверяем каждые 0.2 секунды (как было), но также логируем состояние очереди
        if snow_event is None and WAIT_FOR_SNOW_SECONDS > 0:
            wait_deadline = _now() + timedelta(seconds=min(WAIT_FOR_SNOW_SECONDS, self.window.total_seconds()))
            wait_duration = (wait_deadline - _now()).total_seconds()
            print(f"[MERGER] no snow match yet, waiting up to {wait_duration:.1f}s for late snow...")
            check_count = 0
            while _now() < wait_deadline:
                await asyncio.sleep(0.2)
                check_count += 1
                with self._lock:
                    # ВАЖНО: очистка должна использовать текущее время, а не anpr_time
                    current_time = _now()
                    self._cleanup(current_time)
                    queue_size_before = len(self._snow_events)
                    snow_event = self._pop_match(anpr_time)
                    queue_size_after = len(self._snow_events)
                if snow_event:
                    print(f"[MERGER] found late snow match while waiting (after {check_count * 0.2:.1f}s, {check_count} checks)")
                    # Удаляем ANPR событие из очереди, так как оно будет обработано
                    with self._lock:
                        # Находим и удаляем соответствующее ANPR событие (то, которое было добавлено в очередь)
                        # Ищем по времени и данным события
                        for idx, anpr_evt in enumerate(self._anpr_events):
                            # Сравниваем время и номер для точного совпадения
                            if (abs((anpr_evt.event_time - anpr_time).total_seconds()) < 1.0 and
                                anpr_evt.event_data.get("plate") == anpr_event.get("plate")):
                                del self._anpr_events[idx]
                                print(f"[MERGER] removed matched ANPR event from queue")
                                break
                    break
                # Логируем каждые 5 проверок (раз в секунду), чтобы не спамить
                if check_count % 5 == 0:
                    elapsed = check_count * 0.2
                    remaining = wait_duration - elapsed
                    print(f"[MERGER] still waiting for snow match (elapsed={elapsed:.1f}s, remaining={remaining:.1f}s, queue_size={queue_size_before})")
        
        # Если снег так и не найден, ANPR событие уже в очереди, просто отправляем без снега
        return await self._combine_and_send_internal(anpr_event, detection_bytes, feature_bytes, license_bytes, snow_event, anpr_time)

    async def _combine_and_send_internal(
        self,
        anpr_event: Dict[str, Any],
        detection_bytes: bytes | None,
        feature_bytes: bytes | None,
        license_bytes: bytes | None,
        snow_event: Optional[SnowEvent],
        anpr_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Внутренний метод для объединения и отправки событий.
        """
        combined_event = dict(anpr_event)
        snow_analysis = None

        plate = str(anpr_event.get("plate") or "")
        event_time_iso = str(anpr_event.get("event_time") or "")
        now = _now()

        if self._is_processed(plate, event_time_iso):
            print(f"[MERGER] SKIP duplicate send: plate={plate}, event_time={event_time_iso}")
            return {"sent": False, "status": None, "error": "duplicate_anpr_already_processed"}

        # Помечаем, чтобы второй проход (snow->ANPR или ANPR->snow) не запускал Gemini/отправку повторно
        self._mark_processed(plate, event_time_iso, now)

        if snow_event:
            # Отложенный анализ: вызываем Gemini только когда есть матч с номером
            run_gemini = snow_event.photo_bytes and self._gemini_api_key

            # Если требуется проверка whitelist, делаем её перед Gemini
            vehicle_exists = None
            if self._vehicle_check_url:
                vehicle_exists = await self._check_vehicle_exists(plate)
                if vehicle_exists is False:
                    print(f"[MERGER] whitelist check failed, skip Gemini: plate={plate}")
                    run_gemini = False

            if run_gemini:
                print(f"[MERGER] Running Gemini analysis for matched snow event (photo_size={len(snow_event.photo_bytes)} bytes, bbox={snow_event.payload.get('bbox')})...")
                snow_analysis = self._analyze_snow_gemini(
                    snow_event.photo_bytes, snow_event.payload.get("bbox")
                )
                print(f"[MERGER] Gemini analysis result: {snow_analysis}")
                percentage, confidence = self._extract_snow_fields(snow_analysis)
                print(f"[MERGER] Extracted snow fields: percentage={percentage}, confidence={confidence}")
            else:
                if not snow_event.photo_bytes:
                    print("[MERGER] WARNING: snow_event.photo_bytes is None or empty")
                if not self._gemini_api_key:
                    print("[MERGER] WARNING: GEMINI_API_KEY is not set")
                if self._vehicle_check_url and vehicle_exists is False:
                    print("[MERGER] INFO: vehicle not in whitelist, skipping Gemini and snow fields")
                percentage, confidence = 0.0, 0.0

            combined_event.update(
                {
                    "snow_volume_percentage": percentage if percentage is not None else 0.0,
                    "snow_volume_confidence": max(confidence if confidence is not None else 0.0, 0.05),
                    "matched_snow": True,
                }
            )
            print(f"[MERGER] Final combined event snow fields: percentage={combined_event.get('snow_volume_percentage')}, confidence={combined_event.get('snow_volume_confidence')}")
            if snow_analysis is not None:
                combined_event["snow_gemini_raw"] = snow_analysis
        else:
            # Всегда заполняем поля о снеге, даже если снег не найден
            combined_event.update(
                {
                    "snow_volume_percentage": 0.0,
                    "snow_volume_confidence": 0.0,
                    "matched_snow": False,
                }
            )

        # Если требуется обязательный матч со снегом — не отправляем без matched_snow
        if REQUIRE_SNOW_MATCH and not combined_event.get("matched_snow"):
            result = {
                "sent": False,
                "status": None,
                "error": "snow match required but not found",
                "matched_snow": False,
            }
            print("[MERGER] snow match required, skipping upstream send")
            return result

        # Логируем номер и формат времени перед отправкой
        plate_value = combined_event.get("plate", "N/A")
        event_time_value = combined_event.get("event_time", "N/A")
        print(f"[MERGER] SENDING EVENT - plate: '{plate_value}' (type: {type(plate_value).__name__})")
        print(f"[MERGER] SENDING EVENT - event_time: '{event_time_value}' (type: {type(event_time_value).__name__})")
        print(f"[MERGER] SENDING EVENT - full JSON keys: {list(combined_event.keys())}")
        
        # Проверяем обязательные поля
        required_fields = ["camera_id", "event_time", "plate", "confidence", "direction", "lane", "vehicle"]
        missing_fields = [f for f in required_fields if f not in combined_event]
        if missing_fields:
            print(f"[MERGER] WARNING: missing required fields: {missing_fields}")
        
        data = {"event": json.dumps(combined_event, ensure_ascii=False)}
        
        files = []

        if detection_bytes:
            files.append(
                ("photos", ("detectionPicture.jpg", detection_bytes, "image/jpeg"))
            )
        if feature_bytes:
            files.append(
                ("photos", ("featurePicture.jpg", feature_bytes, "image/jpeg"))
            )
        if license_bytes:
            files.append(
                ("photos", ("licensePlatePicture.jpg", license_bytes, "image/jpeg"))
            )
        if snow_event and snow_event.photo_bytes:
            files.append(
                ("photos", ("snowSnapshot.jpg", snow_event.photo_bytes, "image/jpeg"))
            )

        result = {
            "sent": False,
            "status": None,
            "error": None,
            "matched_snow": bool(snow_event),
        }
        
        # Добавляем данные снега в результат для логирования
        if snow_event:
            result["snow_data"] = {
                "snow_volume_percentage": snow_event.payload.get("snow_volume_percentage"),
                "snow_volume_confidence": snow_event.payload.get("snow_volume_confidence"),
            }
            if "snow_gemini_raw" in snow_event.payload:
                result["snow_data"]["snow_gemini_raw"] = snow_event.payload["snow_gemini_raw"]

        if not self.upstream_url:
            result["error"] = "UPSTREAM_URL is empty"
            print(f"[MERGER] {result['error']}")
            return result

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Если есть файлы - отправляем multipart/form-data
                # Если файлов нет - отправляем как JSON (application/json)
                if files:
                    print(f"[MERGER] sending multipart request with {len(files)} files")
                    print(f"[MERGER] multipart data keys: {list(data.keys())}")
                    resp = await client.post(
                        self.upstream_url,
                        data=data,
                        files=files,
                    )
                else:
                    # Нет файлов - отправляем как JSON
                    print(f"[MERGER] sending JSON request (no files)")
                    print(f"[MERGER] JSON payload size: {len(json.dumps(combined_event, ensure_ascii=False))} bytes")
                    resp = await client.post(
                        self.upstream_url,
                        json=combined_event,
                        headers={"Content-Type": "application/json"},
                    )
            result["sent"] = resp.is_success
            result["status"] = resp.status_code
            
            # Парсим ответ от anpr-service чтобы узнать, была ли машина найдена
            vehicle_exists = None
            if resp.is_success and resp.status_code == 201:
                try:
                    response_json = resp.json()
                    vehicle_exists = response_json.get("vehicle_exists", None)
                except Exception:
                    pass  # Не критично, если не удалось распарсить
            
            # Если машины нет в базе (vehicle_exists = false), возвращаем снеговое событие обратно в очередь
            if snow_event and vehicle_exists is False:
                self.restore_snow_event(snow_event)
                print(
                    f"[MERGER] vehicle not found in database, restored snow event to queue"
                )
            
            if not resp.is_success:
                error_text = resp.text[:400] if resp.text else "No error message"
                result["error"] = error_text
                print(f"[MERGER] ERROR from upstream: status={resp.status_code}, error={error_text}")
            print(
                f"[MERGER] upstream sent={result['sent']} "
                f"status={result['status']} matched_snow={result['matched_snow']} "
                f"vehicle_exists={vehicle_exists}"
            )
        except Exception as e:
            result["error"] = str(e)
            print(f"[MERGER] error while sending event: {e}")
            # При ошибке тоже возвращаем событие снега обратно, чтобы не потерять данные
            if snow_event:
                self.restore_snow_event(snow_event)
                print(f"[MERGER] error occurred, restored snow event to queue")

        return result


_merger_instance: EventMerger | None = None


def init_merger(
    upstream_url: str,
    window_seconds: int = 30,
    ttl_seconds: int = 60,
) -> EventMerger:
    """
    Initialize (or return existing) EventMerger singleton.
    """
    global _merger_instance
    if _merger_instance is None:
        _merger_instance = EventMerger(
            upstream_url=upstream_url,
            window_seconds=window_seconds,
            ttl_seconds=ttl_seconds,
        )
    return _merger_instance
