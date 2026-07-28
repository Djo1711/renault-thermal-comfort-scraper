from src.pipeline.prefilter import KeywordPrefilter

KEYWORDS_YAML = """
categories:
  buee_desembuage:
    - buée
    - désembuage
  chauffage:
    - chauffage
    - "ne chauffe pas"
"""


def make_prefilter(tmp_path):
    path = tmp_path / "keywords.yaml"
    path.write_text(KEYWORDS_YAML, encoding="utf-8")
    return KeywordPrefilter(path)


def test_is_relevant_true_when_keyword_present(tmp_path):
    prefilter = make_prefilter(tmp_path)
    assert prefilter.is_relevant("J'ai de la buée sur le pare-brise") is True


def test_is_relevant_false_when_no_keyword(tmp_path):
    prefilter = make_prefilter(tmp_path)
    assert prefilter.is_relevant("La voiture consomme peu sur autoroute") is False


def test_is_relevant_is_case_insensitive(tmp_path):
    prefilter = make_prefilter(tmp_path)
    assert prefilter.is_relevant("BUÉE partout ce matin") is True


def test_is_relevant_true_on_real_project_keywords():
    prefilter = KeywordPrefilter("config/keywords.yaml")
    assert prefilter.is_relevant("le chauffage ne fonctionne plus depuis hier") is True
    assert prefilter.is_relevant("la peinture s'écaille sur le capot") is False
