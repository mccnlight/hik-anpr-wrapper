from modules.anpr import _select_best_valid_plate


def test_selects_valid_over_invalid():
    trials = [
        ("clahe", "AXINS", None, 0.46),
        ("binary", "809ATT15", "809ATT15", 0.30),
    ]
    plate, conf = _select_best_valid_plate(trials)
    assert plate == "809ATT15"
    assert conf == 0.30


def test_returns_none_when_no_valid():
    trials = [
        ("clahe", "AXINS", None, 0.60),
        ("binary", "J8ATS", None, 0.50),
    ]
    plate, conf = _select_best_valid_plate(trials)
    assert plate is None
    assert conf == 0.0

