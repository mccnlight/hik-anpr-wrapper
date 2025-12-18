from limitations.plate_rules import normalize_primary_plate, is_primary_plate_format


def test_primary_formats_pass():
    assert normalize_primary_plate("809ATT15") == "809ATT15"
    assert normalize_primary_plate("123AB01") == "123AB01"
    assert is_primary_plate_format("456XYZ19") is True  # region 19 → фикс в normalize_primary_plate? actually region set 01-20? 19 ok


def test_invalid_formats_fail():
    assert normalize_primary_plate("AXINS") is None
    assert normalize_primary_plate("12ABC123") is None
    assert is_primary_plate_format("AXINS") is False


def test_suffix_fixed_to_region_15():
    assert normalize_primary_plate("375AA115") == "375AA15"
    assert normalize_primary_plate("375AA155") == "375AA15"
    assert normalize_primary_plate("375AA19") == "375AA15"

