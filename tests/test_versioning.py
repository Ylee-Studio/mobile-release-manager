from src.release_workflow import extract_version_from_text, increment_minor_version


def test_increment_minor_version() -> None:
    assert increment_minor_version("5.100.0") == "5.101.0"


def test_increment_minor_version_uses_default_when_none() -> None:
    assert increment_minor_version(None) == "1.0.0"


def test_extract_version_from_text() -> None:
    assert extract_version_from_text("Запускаем 5.102.0 сегодня") == "5.102.0"
    assert extract_version_from_text("Без версии") is None
