"""Tests for ui.i18n translations."""

from ui.i18n import TRANSLATIONS, get_translator


def test_get_translator_en():
    """English translator returns English text."""
    t = get_translator("en")
    assert t("tab_home") == "Home"
    assert t("tab_search") == "Search & Q&A"


def test_get_translator_ru():
    """Russian translator returns Russian text."""
    t = get_translator("ru")
    assert t("tab_home") == "Главная"
    assert t("tab_search") == "Поиск и Q&A"


def test_unknown_key_returns_key():
    """Unknown key returns the key itself."""
    t = get_translator("en")
    assert t("nonexistent_key_xyz") == "nonexistent_key_xyz"


def test_unknown_lang_falls_back_to_en():
    """Unknown language falls back to English."""
    t = get_translator("fr")
    assert t("tab_home") == "Home"


def test_all_keys_have_both_languages():
    """Every translation key has both 'en' and 'ru' entries."""
    for key, entry in TRANSLATIONS.items():
        assert "en" in entry, f"Missing 'en' for key '{key}'"
        assert "ru" in entry, f"Missing 'ru' for key '{key}'"


def test_no_empty_translations():
    """No translation value is empty."""
    for key, entry in TRANSLATIONS.items():
        for lang, text in entry.items():
            assert text.strip(), f"Empty translation: key='{key}', lang='{lang}'"


def test_translation_count():
    """At least 40 translation keys exist."""
    assert len(TRANSLATIONS) >= 40
