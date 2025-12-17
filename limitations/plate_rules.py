# plate_rules.py
import re
from typing import Optional, Tuple

# Список допустимых регионов (расширишь при необходимости)
REGIONS = {f"{i:02d}" for i in range(1, 21)}  # 01..20


# Паттерны основных форматов KZ
PATTERNS = [
    # Тип 1: 3 цифры + 3 буквы + 2 цифры (регион)
    # 654WOZ05, 850ZEX15 и т.п.
    (
        "type1_3L",
        re.compile(r"^(?P<num>\d{3})(?P<letters>[A-Z]{3})(?P<reg>\d{2})$"),
    ),

    # Тип 1: 3 цифры + 2 буквы + 2 цифры (регион)
    # 389BK01 и т.п.
    (
        "type1_2L",
        re.compile(r"^(?P<num>\d{3})(?P<letters>[A-Z]{2})(?P<reg>\d{2})$"),
    ),

    # Тип 2: грузовые/такси – буква серии + 4 цифры + 2 цифры (регион)
    # A444501, H444501, M123456 и т.п.
    (
        "type2_truck",
        re.compile(r"^(?P<series>[A-Z])(?P<num>\d{4})(?P<reg>\d{2})$"),
    ),

    # Квадратные/короткие: 249AN01 / 249BLM01 / 249ANM01
    # 2–3 цифры + 2–3 буквы + 2 цифры (регион)
    (
        "square_num_letters_reg",
        re.compile(r"^(?P<num>\d{2,3})(?P<letters>[A-Z]{2,3})(?P<reg>\d{2})$"),
    ),

    # Квадратные, где регион слева: 01AN24 / 01ANM24
    (
        "square_reg_letters_num",
        re.compile(r"^(?P<reg>\d{2})(?P<letters>[A-Z]{2,3})(?P<num>\d{2,3})$"),
    ),
]


# Типичные путаницы OCR (символ ↔ символ)
CONFUSION_MAP = {
    "0": "O",
    "O": "0",
    "1": "I",
    "I": "1",
    "5": "S",
    "S": "5",
    "2": "Z",
    "Z": "2",
    "8": "B",
    "B": "8",
}


def clean_ocr_text(text: str) -> str:
    """Приводим строку от OCR к виду A-Z0-9 без пробелов и служебных символов."""
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]", "", text)
    return text


def match_plate(text: str) -> Optional[Tuple[str, re.Match]]:
    """Пробуем сопоставить строку с одним из паттернов гос-номера."""
    for plate_type, pattern in PATTERNS:
        m = pattern.match(text)
        if not m:
            continue
        reg = m.groupdict().get("reg")
        if reg is not None and reg not in REGIONS:
            # регион вне списка – считаем невалидным
            continue
        return plate_type, m
    return None


def is_valid_plate(text: str) -> bool:
    return match_plate(text) is not None


def try_fix_confusions(text: str) -> str:
    """
    Простая эвристика: пробуем заменить ОДИН «путающийся»
    символ (O/0, S/5 и т.п.), чтобы получить валидный номер.
    """
    if is_valid_plate(text):
        return text

    chars = list(text)
    for i, ch in enumerate(chars):
        if ch in CONFUSION_MAP:
            orig = chars[i]
            chars[i] = CONFUSION_MAP[ch]
            candidate = "".join(chars)
            if is_valid_plate(candidate):
                return candidate
            chars[i] = orig  # откат и пробуем дальше

    return text  # ничего не нашли – оставляем как есть


def _normalize_single(text: str) -> Optional[str]:
    """
    Нормализация одной "чистой" строки (только A-Z0-9) без мусора слева/справа.
    Возвращает нормализованный номер или None.
    """
    text = try_fix_confusions(text)

    res = match_plate(text)
    if not res:
        return None

    plate_type, m = res
    g = m.groupdict()

    # Приводим к единым строковым форматам
    if plate_type.startswith("type1"):
        # 3 цифры + 2/3 буквы + 2 цифры
        return f"{g['num']}{g['letters']}{g['reg']}"
    elif plate_type == "type2_truck":
        # буква серии + 4 цифры + 2 цифры
        return f"{g['series']}{g['num']}{g['reg']}"
    elif plate_type == "square_num_letters_reg":
        # 249AN01 / 249ANM01
        return f"{g['num']}{g['letters']}{g['reg']}"
    elif plate_type == "square_reg_letters_num":
        # 01AN24 / 01ANM24 – для единообразия собираем как num+letters+reg
        return f"{g['num']}{g['letters']}{g['reg']}"

    # На всякий случай fallback
    return text


def normalize_plate(raw_text: str) -> Optional[str]:
    """
    На вход – сырая строка от OCR (может содержать мусор слева/справа, типа 'PA654WOZ05').
    На выход – нормализованный номер ('654WOZ05') или None.
    """
    if not raw_text:
        return None

    text = clean_ocr_text(raw_text)
    if not text:
        return None

    # Замена частых OCR-подмен: I/l->1, O/Q->0, B->8, S->5, Z->2
    ocr_fix_map = str.maketrans({
        "I": "1",
        "L": "1",
        "O": "0",
        "Q": "0",
        "B": "8",
        "S": "5",
        "Z": "2",
    })
    text = text.translate(ocr_fix_map)

    # Хак: если распознали 3 цифры + 2/3 буквы + 3 цифры (ошибка OCR типа "346AB155"),
    # считаем, что регион = последние 2 цифры, обрезаем лишнее.
    m = re.match(r'^(\d{3})([A-Z]{2,3})(\d{3})$', text)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)[:2]}"

    # 1) Сначала пробуем всю строку целиком
    candidate = _normalize_single(text)
    if candidate is not None:
        return candidate

    # 2) Если не получилось – ищем номер как ПОДСТРОКУ
    n = len(text)
    # Реальные длины казахских номеров – примерно 6..8 символов.
    MIN_LEN = 6
    MAX_LEN = 9

    best: Optional[str] = None

    for i in range(n):
        for j in range(i + MIN_LEN, min(n, i + MAX_LEN) + 1):
            if j <= i:
                continue
            sub = text[i:j]
            cand = _normalize_single(sub)
            if cand is not None:
                # Берём самый "длинный" кандидат (чаще всего он правильный)
                if best is None or len(cand) > len(best):
                    best = cand

    return best
