from urgenurse.agents.triage import esi_rules


def test_danger_zone_flag_none_when_no_vitals_present() -> None:
    assert esi_rules.danger_zone_flag(None, None, None) is None


def test_danger_zone_flag_false_when_within_adult_range() -> None:
    assert esi_rules.danger_zone_flag(heartrate=84, resprate=16, o2sat=98) is False


def test_danger_zone_flag_true_on_high_adult_heartrate() -> None:
    assert esi_rules.danger_zone_flag(heartrate=140, resprate=16, o2sat=98) is True


def test_danger_zone_flag_true_on_high_adult_resprate() -> None:
    assert esi_rules.danger_zone_flag(heartrate=80, resprate=24, o2sat=98) is True


def test_danger_zone_flag_true_on_low_spo2() -> None:
    assert esi_rules.danger_zone_flag(heartrate=80, resprate=16, o2sat=90) is True


def test_danger_zone_flag_ignores_missing_individual_vitals() -> None:
    # Solo se documentó la FC; dentro de rango adulto -> False, no None.
    assert esi_rules.danger_zone_flag(heartrate=84, resprate=None, o2sat=None) is False


def test_danger_zone_flag_pediatric_threshold_differs_from_adult() -> None:
    # HR=110 es normal para un lactante de 6 meses (<180) pero danger-zone en adulto (>100).
    assert esi_rules.danger_zone_flag(heartrate=110, resprate=None, o2sat=None, age_years=0.5) is False
    assert esi_rules.danger_zone_flag(heartrate=110, resprate=None, o2sat=None, age_years=30) is True


def test_age_years_from_date_of_birth_parses_iso_string() -> None:
    age = esi_rules.age_years_from_date_of_birth("2000-01-01")
    assert 20 < age < 40  # amplio para no depender de la fecha exacta de ejecución


def test_age_years_from_date_of_birth_none_when_missing() -> None:
    assert esi_rules.age_years_from_date_of_birth(None) is None


def test_age_years_from_date_of_birth_none_when_malformed() -> None:
    assert esi_rules.age_years_from_date_of_birth("not-a-date") is None
