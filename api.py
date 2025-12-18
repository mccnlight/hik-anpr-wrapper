from typing import Any, Dict, Optional
import os
import pathlib
import datetime
import json
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse

from modules.anpr import ANPR
from combined_merger import init_merger
from limitations.plate_rules import normalize_primary_plate

app = FastAPI(
    title="Hikvision ANPR Wrapper",
    description="HTTP API для распознавания гос-номеров по кадру камеры",
    version="1.0.0",
)

# создаём движок один раз, чтобы модели не грузились на каждый запрос
engine = ANPR()

# URL внешнего сервиса, куда шлём JSON + фото
UPSTREAM_URL = os.getenv(
    "UPSTREAM_URL", "https://snowops-anpr-service.onrender.com/api/v1/anpr/events"
)
PLATE_CAMERA_ID = os.getenv("PLATE_CAMERA_ID", "camera-001")
MERGE_WINDOW_SECONDS = int(os.getenv("MERGE_WINDOW_SECONDS", "120"))
MERGE_TTL_SECONDS = int(os.getenv("MERGE_TTL_SECONDS", "180"))
ENABLE_SNOW_WORKER = os.getenv("ENABLE_SNOW_WORKER", "false").lower() == "true"
VEHICLE_CHECK_URL = os.getenv("VEHICLE_CHECK_URL")  # опционально: GET ?plate=KZ123ABC -> 200 если есть
VEHICLE_CHECK_TOKEN = os.getenv("VEHICLE_CHECK_TOKEN", "")

merger = init_merger(
    upstream_url=UPSTREAM_URL,
    window_seconds=MERGE_WINDOW_SECONDS,
    ttl_seconds=MERGE_TTL_SECONDS,
)


@app.get("/health", summary="Health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
async def start_background_workers():
    if ENABLE_SNOW_WORKER:
        from snow_worker import start_snow_worker, SNOW_VIDEO_SOURCE_URL

        print(f"[STARTUP] snow worker enabled, source={SNOW_VIDEO_SOURCE_URL}")
        start_snow_worker(UPSTREAM_URL)
    else:
        print("[STARTUP] snow worker disabled (set ENABLE_SNOW_WORKER=true to enable)")


@app.post("/anpr", summary="Recognize Plate")
async def recognize_plate_anpr(
    file: UploadFile = File(..., description="Изображение (JPEG/PNG)"),
) -> JSONResponse:
    """
    Принимает изображение (multipart/form-data, поле: file),
    возвращает JSON с номером и метаданными:

    {
      "plate": "850ZEX15",
      "det_conf": 0.87,
      "ocr_conf": 0.91,
      "bbox": [x1, y1, x2, y2]
    }
    """
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    np_arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image")

    result: Dict[str, Any] = engine.infer(img)
    return JSONResponse(content=result)


# === Работа с файловой структурой для логов ===

BASE_DIR = pathlib.Path("hik_raws")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def get_paths():
    """
    Возвращает:
      time_str  - строка времени для имени файла
      PARTS_DIR - папка для multipart-частей (xml)
      IMAGES_DIR - пока не используем, но создаём на будущее
    """
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")  # 23-59-12

    date_root = BASE_DIR / date_str
    parts_dir = date_root / "parts"
    images_dir = date_root / "images"

    for d in (parts_dir, images_dir):
        d.mkdir(parents=True, exist_ok=True)

    return time_str, parts_dir, images_dir


# === Парсер anpr.xml от Hikvision ===

def parse_anpr_xml(xml_bytes: bytes) -> Dict[str, Any]:
    """
    Парсим anpr.xml от Hikvision.
    Извлекаем:
      - plate / original_plate
      - confidenceLevel (0..1)
      - eventType и dateTime
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return {}

    ns = {"isapi": "http://www.isapi.org/ver20/XMLSchema"}

    def txt(path: str) -> str | None:
        return root.findtext(path, default=None, namespaces=ns)

    event_type = txt("isapi:eventType")
    date_time = txt("isapi:dateTime")

    anpr = root.find("isapi:ANPR", ns)
    if anpr is None:
        return {
            "event_type": event_type,
            "date_time": date_time,
        }

    def txt_anpr(path: str) -> str | None:
        return anpr.findtext(path, default=None, namespaces=ns)

    plate = txt_anpr("isapi:licensePlate") or txt_anpr("isapi:originalLicensePlate")
    original_plate = txt_anpr("isapi:originalLicensePlate")

    conf_str = txt_anpr("isapi:confidenceLevel")
    camera_conf = None
    if conf_str:
        try:
            camera_conf = float(conf_str) / 100.0
        except ValueError:
            camera_conf = None

    # Парсим direction из XML, если он есть
    direction = txt_anpr("isapi:direction") or txt("isapi:direction")
    
    # Парсим lane из XML, если он есть
    lane_str = txt_anpr("isapi:lane") or txt("isapi:lane")
    lane = None
    if lane_str:
        try:
            lane = int(lane_str)
        except ValueError:
            lane = None

    result = {
        "event_type": event_type,
        "date_time": date_time,
        "plate": plate,
        "original_plate": original_plate,
        "confidence": camera_conf,
    }
    
    if direction:
        result["direction"] = direction
    if lane is not None:
        result["lane"] = lane
    
    return result


# === Дополнительная проверка номера по БД (опционально) ===
async def check_vehicle_exists(normalized_plate: str) -> Optional[bool]:
    """
    Возвращает True/False если удалось проверить, None если проверка отключена или упала.
    Ожидается эндпоинт VEHICLE_CHECK_URL, принимающий query ?plate=... и отдающий 2xx если номер найден.
    """
    if not VEHICLE_CHECK_URL or not normalized_plate:
        return None

    try:
        headers = {}
        if VEHICLE_CHECK_TOKEN:
            headers["Authorization"] = f"Bearer {VEHICLE_CHECK_TOKEN}"

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(VEHICLE_CHECK_URL, params={"plate": normalized_plate}, headers=headers)
            print(f"[VEHICLE_CHECK] plate={normalized_plate} status={resp.status_code}")
            if resp.status_code >= 500:
                return None
            return resp.status_code < 300
    except Exception as e:
        print(f"[VEHICLE_CHECK] failed for plate={normalized_plate}: {e}")
        return None


async def choose_plate(camera_plate: str | None, model_plate: str | None) -> tuple[Optional[str], str]:
    """
    Приоритет: валидный номер камеры → валидный номер модели.
    Если включена VEHICLE_CHECK_URL, пытаемся подтвердить существование в базе.
    Возвращает (plate_or_none, reason).
    """
    cam_norm = normalize_primary_plate(camera_plate) if camera_plate else None
    model_norm = normalize_primary_plate(model_plate) if model_plate else None

    # Сначала камера: если валидна и подтверждена БД — берем сразу
    if cam_norm:
        exists = await check_vehicle_exists(cam_norm)
        if exists is True:
            return cam_norm, "camera_valid_db_match"
        if exists is None:
            # проверка недоступна — берем валидную камеру
            return cam_norm, "camera_valid_no_db"
        # exists False — попробуем модель

    if model_norm:
        exists = await check_vehicle_exists(model_norm)
        if exists is True:
            return model_norm, "model_valid_db_match"
        if exists is None and cam_norm is None:
            return model_norm, "model_valid_no_db"

    if cam_norm:
        return cam_norm, "camera_valid_fallback"
    if model_norm:
        return model_norm, "model_valid_fallback"
    return None, "no_valid_plate"


# === Отправка события и фотографий на внешний сервис ===

async def send_to_upstream(
    event_data: Dict[str, Any],
    detection_bytes: bytes | None,
    feature_bytes: bytes | None,
    license_bytes: bytes | None,
) -> Dict[str, Any]:
    """
    Отправка события и фотографий на внешний ANPR-сервис.

    Формат, ожидаемый бэкендом:
      Content-Type: multipart/form-data

      Поля формы:
        - event  (обязательное)  — JSON-строка с данными события
        - photos (опционально)   — файлы фотографий, одно или несколько полей photos

    Возвращает:
      {
        "sent": bool,
        "status": int | None,
        "error": str | None,
      }
    """
    if not UPSTREAM_URL:
        msg = "UPSTREAM_URL is not set"
        print(f"[UPSTREAM] {msg}")
        return {
            "sent": False,
            "status": None,
            "error": msg,
        }

    try:
        # event — как строка JSON в поле формы
        event_str = json.dumps(event_data, ensure_ascii=False)
        data = {
            "event": event_str,
        }

        print("[UPSTREAM] EVENT JSON:")
        print(event_str)

        # photos — список файлов под одним и тем же ключом "photos"
        files = []

        if detection_bytes:
            print(
                f"[UPSTREAM] add photo: field='photos', name='detectionPicture.jpg', "
                f"size={len(detection_bytes)}"
            )
            files.append(
                (
                    "photos",
                    ("detectionPicture.jpg", detection_bytes, "image/jpeg"),
                )
            )

        if feature_bytes:
            print(
                f"[UPSTREAM] add photo: field='photos', name='featurePicture.jpg', "
                f"size={len(feature_bytes)}"
            )
            files.append(
                (
                    "photos",
                    ("featurePicture.jpg", feature_bytes, "image/jpeg"),
                )
            )

        if license_bytes:
            print(
                f"[UPSTREAM] add photo: field='photos', name='licensePlatePicture.jpg', "
                f"size={len(license_bytes)}"
            )
            files.append(
                (
                    "photos",
                    ("licensePlatePicture.jpg", license_bytes, "image/jpeg"),
                )
            )

        async with httpx.AsyncClient(timeout=10.0) as client:
            # data + files => multipart/form-data
            resp = await client.post(UPSTREAM_URL, data=data, files=files or None)
            print(f"[UPSTREAM] status={resp.status_code}, body={resp.text[:400]}")

            return {
                "sent": resp.is_success,
                "status": resp.status_code,
                "error": None if resp.is_success else resp.text[:400],
            }

    except Exception as e:
        # Не ломаем обработку камеры, просто логируем
        print(f"[UPSTREAM] error while sending event: {e}")
        return {
            "sent": False,
            "status": None,
            "error": str(e),
        }


# === Основной хендлер для Hikvision ANPR ===

@app.post("/api/v1/anpr/hikvision")
async def hikvision_isapi(request: Request):
    # Директории вида hik_raws/YYYY-MM-DD/{parts,images}
    time_str, PARTS_DIR, IMAGES_DIR = get_paths()
    headers = dict(request.headers)

    print("=== HIKVISION REQUEST HEADERS ===")
    for k, v in headers.items():
        print(f"{k}: {v}")

    body = await request.body()
    content_type = headers.get("content-type", "")

    # Информация камеры (из anpr.xml)
    camera_info: Dict[str, Any] = {}
    camera_xml_path: str | None = None

    # Результат нашего ANPR по detectionPicture.jpg
    model_plate: str | None = None
    model_det_conf: float | None = None
    model_ocr_conf: float | None = None
    model_bbox: Any = None

    # Байты трёх картинок (их отправим дальше, но не сохраняем на диск)
    detection_bytes: bytes | None = None
    feature_bytes: bytes | None = None
    license_bytes: bytes | None = None

    # === ВАРИАНТ 1: multipart/form-data (основной для Hikvision) ===
    if "multipart/form-data" in content_type:
        form = await request.form()
        found_files = 0

        for key, value in form.items():
            if hasattr(value, "filename"):
                found_files += 1
                file_bytes = await value.read()
                fname = value.filename or f"hik_file_{time_str}_{found_files}.bin"
                ftype = value.content_type or "application/octet-stream"

                print(
                    f"[HIK] part field={key}, name={fname}, "
                    f"type={ftype}, size={len(file_bytes)}"
                )

                # anpr.xml → номер камеры, сохраняем XML на диск
                if fname.lower().endswith("anpr.xml"):
                    part_path = PARTS_DIR / f"{time_str}_{fname}"
                    part_path.write_bytes(file_bytes)
                    camera_xml_path = str(part_path)
                    camera_info = parse_anpr_xml(file_bytes)
                    # Логируем номер от камеры сразу после парсинга
                    camera_plate_detected = camera_info.get("plate")
                    if camera_plate_detected:
                        print(f"[HIK] CAMERA DETECTED PLATE: '{camera_plate_detected}' (from anpr.xml)")
                    continue

                # Картинки держим в памяти, на диск не кладём
                if ftype.startswith("image/"):
                    lower_name = fname.lower()

                    if lower_name == "detectionpicture.jpg":
                        detection_bytes = file_bytes

                        # Гоняем через наш ANPR
                        np_arr = np.frombuffer(file_bytes, np.uint8)
                        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            anpr_res = engine.infer(img)
                            model_plate = anpr_res.get("plate")
                            model_det_conf = anpr_res.get("det_conf")
                            model_ocr_conf = anpr_res.get("ocr_conf")
                            model_bbox = anpr_res.get("bbox")
                            # Логируем номер от модели сразу после распознавания
                            if model_plate:
                                print(f"[HIK] MODEL DETECTED PLATE: '{model_plate}' (det_conf={model_det_conf}, ocr_conf={model_ocr_conf})")

                    elif lower_name == "featurepicture.jpg":
                        feature_bytes = file_bytes

                    elif lower_name == "licenseplatepicture.jpg":
                        license_bytes = file_bytes

            else:
                # текстовые части (если будут) — просто логируем
                print(f"[HIK] form field: {key} = {value}")

        # === Формируем JSON-событие в простом виде ===

        # Используем UTC timezone для RFC3339 формата (требуется Go)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        now_iso = now_utc.isoformat().replace('+00:00', 'Z')  # RFC3339 формат

        camera_plate = camera_info.get("plate")
        camera_conf = camera_info.get("confidence")
        
        # Парсим event_time из камеры или используем текущее время
        event_time_str = camera_info.get("date_time")
        if event_time_str:
            # Пытаемся распарсить время от камеры
            try:
                # Если камера отправляет без timezone, добавляем UTC
                if 'Z' not in event_time_str and '+' not in event_time_str:
                    event_time_str = event_time_str + 'Z'
                event_time = event_time_str
            except Exception:
                event_time = now_iso
        else:
            event_time = now_iso

        # Выбираем основной номер: камера приоритетна, модель — только если валидна.
        main_plate, plate_reason = await choose_plate(camera_plate, model_plate)
        print(
            f"[HIK] PLATE CHOICE: main='{main_plate}' reason='{plate_reason}' "
            f"(camera='{camera_plate}', model='{model_plate}')"
        )

        # Проверка: есть ли anpr.xml от камеры
        has_anpr_xml = camera_xml_path is not None

        # Проверка валидности номера (строгий формат)
        has_valid_plate = bool(main_plate)

        # Проверка уверенности:
        # - если берем камеру (reason содержит "camera"), полагаемся на camera_conf >= 0.9
        # - если берем модель, требуем det_conf >= 0.3 и ocr_conf >= 0.5
        plate_from_camera = plate_reason.startswith("camera") and camera_conf is not None
        plate_from_model = plate_reason.startswith("model")

        has_valid_confidence = False
        if plate_from_camera:
            # Снижаем порог доверия камеры: 0.75 достаточно, если формат валиден.
            # Если конфиденс отсутствует, но формат валиден — считаем пригодным.
            if camera_conf is None:
                has_valid_confidence = True
            else:
                has_valid_confidence = camera_conf >= 0.75
        elif plate_from_model:
            if model_det_conf is not None and model_ocr_conf is not None:
                has_valid_confidence = model_det_conf >= 0.3 and model_ocr_conf >= 0.5

        # Если нет валидного номера или уверенности — не отправляем
        if not has_valid_plate or not has_valid_confidence:
            print(
                f"[HIK] SKIPPING EVENT: plate='{main_plate}' reason='{plate_reason}' "
                f"det_conf={model_det_conf} ocr_conf={model_ocr_conf} camera_conf={camera_conf}"
            )
            log_event = {
                "timestamp": now_iso,
                "kind": "skipped_invalid_or_low_conf",
                "has_anpr_xml": has_anpr_xml,
                "plate": main_plate,
                "plate_reason": plate_reason,
                "model_plate": model_plate,
                "camera_plate": camera_plate,
                "model_det_conf": model_det_conf,
                "model_ocr_conf": model_ocr_conf,
                "camera_conf": camera_conf,
                "upstream_sent": False,
                "upstream_status": None,
                "upstream_error": "skipped: invalid format or low confidence",
            }
            log_path = BASE_DIR / "detections.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_event, ensure_ascii=False) + "\n")
            return JSONResponse({"status": "ok"})

        effective_conf = 0.0
        if plate_from_camera and camera_conf is not None:
            effective_conf = float(camera_conf)
        elif plate_from_model and model_ocr_conf is not None:
            effective_conf = float(model_ocr_conf)

        event_data: Dict[str, Any] = {
            # контракт бэкенда (обязательные поля)
            "camera_id": PLATE_CAMERA_ID,  # можно сконфигурировать через env
            "event_time": event_time,  # RFC3339 формат с timezone
            "plate": main_plate,
            "confidence": effective_conf,  # обязательное поле
            "direction": camera_info.get("direction", "unknown"),  # обязательное поле
            "lane": int(camera_info.get("lane", 0)),  # обязательное поле
            "vehicle": {},  # обязательное поле (пустой объект если нет данных)

            # понятные поля
            "camera_plate": camera_plate,
            "camera_confidence": camera_conf,
            "model_plate": model_plate,
            "model_det_conf": model_det_conf,
            "model_ocr_conf": model_ocr_conf,
            "plate_source": plate_reason,
            "xml_event_type": camera_info.get("event_type"),  # тип события из anpr.xml

            # доп. служебное время
            "timestamp": now_iso,
        }

        # 5) Отправляем JSON + фото на внешний сервис и получаем результат
        # Логируем, какие файлы есть
        files_info = []
        if detection_bytes:
            files_info.append(f"detection({len(detection_bytes)} bytes)")
        if feature_bytes:
            files_info.append(f"feature({len(feature_bytes)} bytes)")
        if license_bytes:
            files_info.append(f"license({len(license_bytes)} bytes)")
        print(f"[HIK] files to send: {files_info if files_info else 'NONE'}")
        
        # Фильтрация событий от уезжающих машин
        # Проверяем direction и event_type, чтобы отфильтровать выезжающие машины
        direction = event_data.get("direction", "").lower()
        event_type = camera_info.get("event_type", "").lower() if camera_info else ""
        
        # Фильтруем события, если:
        # 1. direction указывает на выезд (например, "out", "exit", "leaving", "departure")
        # 2. event_type указывает на выезд (например, "vehicleexit", "departure")
        # 3. direction содержит "left" или "right" в контексте выезда (но не "left lane" или "right lane")
        is_exiting = False
        exit_keywords = ["out", "exit", "leaving", "departure", "away"]
        
        if direction and any(keyword in direction for keyword in exit_keywords):
            is_exiting = True
            print(f"[HIK] FILTERED: event filtered - direction='{direction}' indicates vehicle is exiting")
        
        if event_type and any(keyword in event_type for keyword in exit_keywords):
            is_exiting = True
            print(f"[HIK] FILTERED: event filtered - event_type='{event_type}' indicates vehicle is exiting")
        
        if is_exiting:
            # Логируем отфильтрованное событие
            log_event = {
                **event_data,
                "upstream_sent": False,
                "upstream_status": None,
                "upstream_error": "filtered: vehicle is exiting",
                "matched_snow": False,
                "filtered_reason": f"direction='{direction}', event_type='{event_type}'",
            }
            log_path = BASE_DIR / "detections.log"
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_event, ensure_ascii=False) + "\n")
            print(f"[HIK] Event filtered and logged, not sending to upstream")
            return JSONResponse({"status": "ok"})
        
        upstream_result = await merger.combine_and_send(
            anpr_event=event_data,
            detection_bytes=detection_bytes,
            feature_bytes=feature_bytes,
            license_bytes=license_bytes,
        )

        # 6) Логируем в detections.log (включая статус отправки и данные снега)
        log_event = {
            **event_data,
            "upstream_sent": upstream_result["sent"],
            "upstream_status": upstream_result["status"],
            "upstream_error": upstream_result["error"],
            "matched_snow": upstream_result.get("matched_snow"),
        }
        
        # Добавляем данные снега в лог, если они есть
        if "snow_data" in upstream_result:
            log_event.update(upstream_result["snow_data"])

        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

        # 7) Ответ камере — просто "ok"
        return JSONResponse({"status": "ok"})

    # === ВАРИАНТ 2: fallback — ищем JPEG прямо в body (редкий случай) ===
    start = body.find(b"\xff\xd8\xff")
    end = body.find(b"\xff\xd9", start + 2)

    if start == -1 or end == -1:
        # вообще нет jpeg — логируем, камере просто "ok"
        log_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "kind": "no_jpeg_in_body",
            "body_size": len(body),
            "upstream_sent": False,
            "upstream_status": None,
            "upstream_error": "no_jpeg_in_body",
        }
        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

        return JSONResponse({"status": "ok"})

    jpg_bytes = body[start: end + 2]
    np_arr = np.frombuffer(jpg_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        # Декод не удался — логируем, камере отдаём ошибку
        log_event = {
            "timestamp": datetime.datetime.now().isoformat(),
            "kind": "jpeg_decode_error",
            "upstream_sent": False,
            "upstream_status": None,
            "upstream_error": "cannot_decode_jpeg",
        }
        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

        return JSONResponse(
            {"status": "error", "message": "cannot decode jpeg"}, status_code=400
        )

    anpr_res = engine.infer(img)
    model_plate = anpr_res.get("plate")
    model_det_conf = anpr_res.get("det_conf")
    model_ocr_conf = anpr_res.get("ocr_conf")
    model_bbox = anpr_res.get("bbox")
    
    # Логируем номер от модели сразу после распознавания (fallback вариант)
    if model_plate:
        print(f"[HIK] MODEL DETECTED PLATE (fallback): '{model_plate}' (det_conf={model_det_conf}, ocr_conf={model_ocr_conf})")

    # Используем UTC timezone для RFC3339 формата (требуется Go)
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    now_iso = now_utc.isoformat().replace('+00:00', 'Z')  # RFC3339 формат

    # тут камеры нет, только модель
    main_plate = model_plate
    
    # Логируем итоговый номер (fallback вариант)
    if main_plate:
        print(f"[HIK] FINAL PLATE (fallback): '{main_plate}'")

    # Проверка: есть ли нормальный номер (не None и не пустая строка)
    has_valid_plate = main_plate and main_plate.strip() and main_plate != "unknown"
    
    # Проверка уверенности: модель должна иметь достаточную уверенность
    # det_conf >= 0.3 и ocr_conf >= 0.5
    has_valid_confidence = False
    if model_det_conf is not None and model_ocr_conf is not None:
        has_valid_confidence = model_det_conf >= 0.3 and model_ocr_conf >= 0.5
    
    # Если нет нормального номера ИЛИ нет достаточной уверенности - не отправляем событие
    # Это предотвращает отправку событий с плохо распознанными номерами
    if not has_valid_plate or not has_valid_confidence:
        print(f"[HIK] SKIPPING EVENT (fallback): plate='{main_plate}', det_conf={model_det_conf}, ocr_conf={model_ocr_conf}")
        # Логируем пропущенное событие
        log_event = {
            "timestamp": now_iso,
            "kind": "skipped_fallback_no_valid_data",
            "plate": main_plate,
            "model_det_conf": model_det_conf,
            "model_ocr_conf": model_ocr_conf,
            "upstream_sent": False,
            "upstream_status": None,
            "upstream_error": "skipped: no valid plate and low confidence",
        }
        log_path = BASE_DIR / "detections.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_event, ensure_ascii=False) + "\n")
        return JSONResponse({"status": "ok"})

    event_data: Dict[str, Any] = {
        # контракт бэкенда (обязательные поля)
        "camera_id": PLATE_CAMERA_ID,
        "event_time": now_iso,  # RFC3339 формат с timezone
        "plate": main_plate,
        "confidence": float(model_ocr_conf) if model_ocr_conf is not None else 0.0,  # обязательное поле
        "direction": "unknown",  # обязательное поле
        "lane": 0,  # обязательное поле
        "vehicle": {},  # обязательное поле (пустой объект если нет данных)
        
        "camera_plate": None,
        "camera_confidence": None,
        "model_plate": model_plate,
        "model_det_conf": model_det_conf,
        "model_ocr_conf": model_ocr_conf,

        "timestamp": now_iso,
    }

    # Отправляем только одну картинку как detection_picture
    upstream_result = await merger.combine_and_send(
        anpr_event=event_data,
        detection_bytes=jpg_bytes,
        feature_bytes=None,
        license_bytes=None,
    )

    log_event = {
        **event_data,
        "upstream_sent": upstream_result["sent"],
        "upstream_status": upstream_result["status"],
        "upstream_error": upstream_result["error"],
        "anpr_bbox": model_bbox,
        "matched_snow": upstream_result.get("matched_snow"),
    }
    
    # Добавляем данные снега в лог, если они есть
    if "snow_data" in upstream_result:
        log_event.update(upstream_result["snow_data"])

    log_path = BASE_DIR / "detections.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_event, ensure_ascii=False) + "\n")

    # Ответ камере — снова просто "ok"
    return JSONResponse({"status": "ok"})
