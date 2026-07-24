from datetime import datetime, timezone

from click.testing import CliRunner

import main
from src.pipeline.llm_classifier import LLMResponseInvalidError, OllamaUnavailableError
from src.storage.models import ClassifiedPost, LLMClassificationResult, RawPost, VehicleInfo
from src.storage.writer import append_posts, read_posts


def make_classified_post(post_id: str) -> ClassifiedPost:
    return ClassifiedPost(
        id=post_id,
        source="automobile_propre",
        text="texte",
        url=f"https://x/{post_id}",
        scraped_at=datetime.now(timezone.utc),
        published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
        pertinent_confort_thermique=True,
        pertinence_confidence="haute",
        vehicule_marque="Renault",
        vehicule_modele="Zoe",
        sentiment=3,
        sentiment_confidence="haute",
        classification_echec=False,
    )


def make_raw_post(post_id: str, text: str) -> RawPost:
    return RawPost(
        id=post_id,
        source="automobile_propre",
        text=text,
        url=f"https://x/{post_id}",
        scraped_at=datetime.now(timezone.utc),
        published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
    )


def make_llm_result() -> LLMClassificationResult:
    return LLMClassificationResult(
        pertinent_confort_thermique=True,
        pertinence_confidence="haute",
        vehicule=VehicleInfo(),
        type_probleme=["chauffage"],
        type_probleme_confidence="haute",
        sentiment=3,
        sentiment_confidence="haute",
    )


def make_fake_llm_classifier(classify_fn):
    """Builds a stub class usable as a drop-in replacement for main.LLMClassifier.

    `classify_fn(post)` is invoked for each call to `.classify(post)` on the
    instance; whatever it returns (or raises) becomes the result of that call.
    """

    class FakeLLMClassifier:
        def __init__(self, *args, **kwargs):
            pass

        def classify(self, post):
            return classify_fn(post)

    return FakeLLMClassifier


def test_cli_registers_all_three_commands():
    assert set(main.cli.commands.keys()) == {"scrape", "classify", "export"}


def test_export_command_writes_output_files(tmp_path, monkeypatch):
    classified_path = tmp_path / "posts_classified.parquet"
    output_dir = tmp_path / "output"
    append_posts(classified_path, [make_classified_post("1"), make_classified_post("2")])

    monkeypatch.setitem(main.SETTINGS["paths"], "classified_posts", str(classified_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "output_dir", str(output_dir))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["export"])

    assert result.exit_code == 0
    assert (output_dir / "verbatims_classifies.parquet").exists()
    assert (output_dir / "stats_marque_modele_sentiment.parquet").exists()


def make_fake_automobile_propre_scraper(scrape_topic_fn):
    """Builds a stub class usable as a drop-in replacement for
    main.AutomobilePropreScraper, exposing two topics discovered under a single
    brand/model. `scrape_topic_fn(topic_url)` is invoked for each topic and its
    return value (or raised exception) becomes the result of `scrape_topic`.
    """

    class FakeAutomobilePropreScraper:
        def __init__(self, *args, **kwargs):
            pass

        def discover_brand_forums(self):
            return {"Renault": "https://x/forums/renault"}

        def discover_subforums(self, brand_url):
            return {"Zoe": "https://x/forums/renault/zoe"}

        def iter_topic_urls(self, subforum_url, scope):
            yield "https://x/topic/1"
            yield "https://x/topic/2"

        def scrape_topic(self, topic_url, brand, model):
            return scrape_topic_fn(topic_url)

    return FakeAutomobilePropreScraper


def test_scrape_writes_posts_incrementally_surviving_mid_run_crash(tmp_path, monkeypatch):
    """Regression test for the data-loss bug: previously all scraped posts were
    accumulated in memory for the whole run and written to disk only once at the
    end. If the run crashed partway through, everything scraped so far was lost
    even though the checkpoint had already marked those topic URLs as seen. This
    test scrapes two topics where the second one raises, and asserts the first
    topic's post was still persisted to disk despite the crash.
    """
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    calls = []

    def scrape_topic_fn(topic_url):
        calls.append(topic_url)
        if topic_url == "https://x/topic/2":
            raise RuntimeError("simulated crash after topic 1 was scraped")
        return [make_raw_post("t1", "le chauffage ne fonctionne plus")]

    monkeypatch.setattr(
        main, "AutomobilePropreScraper", make_fake_automobile_propre_scraper(scrape_topic_fn)
    )

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "automobile_propre"])

    assert calls == ["https://x/topic/1", "https://x/topic/2"]
    assert result.exception is not None

    # Despite the crash on topic 2, topic 1's post must already be on disk.
    saved = read_posts(raw_path, RawPost)
    assert [p.id for p in saved] == ["t1"]


def test_classify_prefilters_before_calling_llm(tmp_path, monkeypatch):
    raw_path = tmp_path / "posts_raw.parquet"
    classified_path = tmp_path / "posts_classified.parquet"
    relevant_post = make_raw_post("relevant", "le chauffage ne fonctionne plus")
    irrelevant_post = make_raw_post("irrelevant", "la peinture est belle")
    append_posts(raw_path, [relevant_post, irrelevant_post])

    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "classified_posts", str(classified_path))

    calls = []

    def classify_fn(post):
        calls.append(post.id)
        return make_llm_result()

    monkeypatch.setattr(main, "LLMClassifier", make_fake_llm_classifier(classify_fn))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["classify"])

    assert result.exit_code == 0
    assert calls == ["relevant"]


def test_classify_dedups_against_classified_path(tmp_path, monkeypatch):
    raw_path = tmp_path / "posts_raw.parquet"
    classified_path = tmp_path / "posts_classified.parquet"
    already_done = make_raw_post("done", "le chauffage ne fonctionne plus")
    remaining = make_raw_post("remaining", "la climatisation est en panne")
    append_posts(raw_path, [already_done, remaining])
    append_posts(classified_path, [make_classified_post("done")])

    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "classified_posts", str(classified_path))

    calls = []

    def classify_fn(post):
        calls.append(post.id)
        return make_llm_result()

    monkeypatch.setattr(main, "LLMClassifier", make_fake_llm_classifier(classify_fn))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["classify"])

    assert result.exit_code == 0
    assert calls == ["remaining"]


def test_classify_aborts_and_saves_partial_progress_on_ollama_unavailable(tmp_path, monkeypatch):
    raw_path = tmp_path / "posts_raw.parquet"
    classified_path = tmp_path / "posts_classified.parquet"
    first_post = make_raw_post("first", "le chauffage ne fonctionne plus")
    second_post = make_raw_post("second", "la climatisation est en panne")
    append_posts(raw_path, [first_post, second_post])

    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "classified_posts", str(classified_path))

    calls = []

    def classify_fn(post):
        calls.append(post.id)
        if len(calls) == 1:
            return make_llm_result()
        raise OllamaUnavailableError("Ollama injoignable")

    monkeypatch.setattr(main, "LLMClassifier", make_fake_llm_classifier(classify_fn))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["classify"])

    assert result.exit_code != 0
    assert calls == ["first", "second"]
    saved = read_posts(classified_path, ClassifiedPost)
    assert [p.id for p in saved] == ["first"]


def test_classify_marks_invalid_llm_response_failed_and_continues(tmp_path, monkeypatch):
    raw_path = tmp_path / "posts_raw.parquet"
    classified_path = tmp_path / "posts_classified.parquet"
    first_post = make_raw_post("first", "le chauffage ne fonctionne plus")
    second_post = make_raw_post("second", "la climatisation est en panne")
    append_posts(raw_path, [first_post, second_post])

    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "classified_posts", str(classified_path))

    calls = []

    def classify_fn(post):
        calls.append(post.id)
        if len(calls) == 1:
            raise LLMResponseInvalidError("réponse invalide")
        return make_llm_result()

    monkeypatch.setattr(main, "LLMClassifier", make_fake_llm_classifier(classify_fn))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["classify"])

    assert result.exit_code == 0
    assert calls == ["first", "second"]
    saved = read_posts(classified_path, ClassifiedPost)
    assert len(saved) == 2
    by_id = {p.id: p for p in saved}
    assert by_id["first"].classification_echec is True
    assert by_id["second"].classification_echec is False
