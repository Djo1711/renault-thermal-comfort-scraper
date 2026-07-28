import unicodedata


def normalize_text(s: str) -> str:
    """Strip diacritics and lowercase a string for comparison purposes.

    Used so that e.g. "Citroën" (as written in config/vehicle_list.yaml,
    with the diaeresis) matches ASCII-only variants such as the URL slug
    "citroen" (caradisiac.py) or a forum heading rendered without the
    diaeresis (automobile_propre.py), instead of silently never matching.
    """
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
