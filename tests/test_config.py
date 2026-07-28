from src.utils.config import load_yaml


def test_load_yaml_returns_dict(tmp_path):
    p = tmp_path / "sample.yaml"
    p.write_text("a: 1\nb: two\n", encoding="utf-8")
    data = load_yaml(p)
    assert data == {"a": 1, "b": "two"}


def test_load_yaml_reads_project_settings():
    data = load_yaml("config/settings.yaml")
    assert data["rate_limit"]["min_seconds"] == 3
    assert data["ollama"]["model"] == "qwen2.5:7b-instruct"
