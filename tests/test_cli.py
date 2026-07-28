from datetime import datetime, timezone

from click.testing import CliRunner

import main
from src.pipeline.llm_classifier import LLMResponseInvalidError, OllamaUnavailableError
from src.scrapers.base import FetchError
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


def make_fake_automobile_propre_scraper(scrape_topic_fn, direct_topic_urls=None):
    """Builds a stub class usable as a drop-in replacement for
    main.AutomobilePropreScraper, exposing two topics discovered under a single
    brand/model sub-forum. `scrape_topic_fn(topic_url)` is invoked for each
    topic and its return value (or raised exception) becomes the result of
    `scrape_topic`. `direct_topic_urls`, if given, are the topics returned when
    `iter_topic_urls` is called directly on the brand's own forum URL (Finding
    4's brand-level-topics path) -- defaults to none, matching the common case
    of no topics posted directly under the brand.
    """
    direct_topic_urls = direct_topic_urls or []

    class FakeAutomobilePropreScraper:
        def __init__(self, *args, **kwargs):
            pass

        def discover_brand_forums(self):
            return {"Renault": "https://x/forums/renault"}

        def discover_subforums(self, brand_url):
            return {"Zoe": "https://x/forums/renault/zoe"}

        def iter_topic_urls(self, subforum_url, scope):
            if subforum_url == "https://x/forums/renault":
                yield from direct_topic_urls
                return
            yield "https://x/topic/1"
            yield "https://x/topic/2"

        def scrape_topic(self, topic_url, brand, model):
            return scrape_topic_fn(topic_url)

    return FakeAutomobilePropreScraper


def test_scrape_writes_posts_incrementally_surviving_mid_run_crash(tmp_path, monkeypatch):
    """Regression test for the data-loss bug: previously all scraped posts were
    accumulated in memory for the whole run and written to disk only once at the
    end. If the run crashed partway through, everything scraped so far was lost
    even though the checkpoint had already marked those topic URLs as seen.

    Since Finding B (round 2) broadened the per-topic try/except to catch any
    exception raised by a single topic's own scrape_topic() call -- not just
    FetchError -- an exception from *that* call site is no longer a run-ending
    crash. To still exercise a genuine unrecoverable crash (an unexpected bug
    that is NOT scoped to one item, e.g. one raised by the topic-discovery
    generator itself, which only the FetchError guard around it can catch),
    this test has one topic scraped and persisted successfully, then has
    iter_topic_urls raise a plain RuntimeError instead of yielding a second
    topic. That exception is not a FetchError, so it is not caught anywhere,
    and the whole run aborts -- but the already-scraped topic's post must
    still be on disk.
    """
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    calls = []

    class CrashingScraper:
        def __init__(self, *args, **kwargs):
            pass

        def discover_brand_forums(self):
            return {"Renault": "https://x/forums/renault"}

        def discover_subforums(self, brand_url):
            return {"Zoe": "https://x/forums/renault/zoe"}

        def iter_topic_urls(self, subforum_url, scope):
            if subforum_url != "https://x/forums/renault/zoe":
                return
            yield "https://x/topic/1"
            raise RuntimeError("simulated bug in pagination after topic 1 was yielded")

        def scrape_topic(self, topic_url, brand, model):
            calls.append(topic_url)
            return [make_raw_post("t1", "le chauffage ne fonctionne plus")]

    monkeypatch.setattr(main, "AutomobilePropreScraper", CrashingScraper)

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "automobile_propre"])

    assert calls == ["https://x/topic/1"]
    assert result.exception is not None

    # Despite the crash, topic 1's post must already be on disk.
    saved = read_posts(raw_path, RawPost)
    assert [p.id for p in saved] == ["t1"]


def test_scrape_automobile_propre_also_scrapes_topics_posted_directly_under_brand(
    tmp_path, monkeypatch
):
    """Regression test for Finding 4: topics posted directly at the brand-forum
    level (no model sub-forum) were never scraped -- main.py's orchestration
    only ever called iter_topic_urls on discovered sub-forums. This asserts
    that a topic yielded directly on the brand's own forum URL is also
    scraped and persisted, in addition to the sub-forum's topics."""
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    def scrape_topic_fn(topic_url):
        if topic_url == "https://x/topic/direct":
            return [make_raw_post("direct1", "probleme de chauffage direct sous la marque")]
        return [make_raw_post(f"t-{topic_url[-1]}", "la climatisation ne fonctionne plus")]

    monkeypatch.setattr(
        main,
        "AutomobilePropreScraper",
        make_fake_automobile_propre_scraper(
            scrape_topic_fn, direct_topic_urls=["https://x/topic/direct"]
        ),
    )

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "automobile_propre"])

    assert result.exit_code == 0
    assert result.exception is None

    saved_ids = {p.id for p in read_posts(raw_path, RawPost)}
    assert "direct1" in saved_ids


def test_scrape_skips_topic_raising_fetcherror_and_continues_with_others(tmp_path, monkeypatch):
    """Regression test: a FetchError raised by scrape_topic (after base.py's own
    retries are exhausted) must be logged and skipped, not abort the whole
    scrape run. The first topic fails with FetchError, but the second topic
    must still be scraped and its post persisted, with a clean exit."""
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    calls = []

    def scrape_topic_fn(topic_url):
        calls.append(topic_url)
        if topic_url == "https://x/topic/1":
            raise FetchError("échec après 3 tentatives")
        return [make_raw_post("t2", "la climatisation ne refroidit plus")]

    monkeypatch.setattr(
        main, "AutomobilePropreScraper", make_fake_automobile_propre_scraper(scrape_topic_fn)
    )

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "automobile_propre"])

    assert result.exit_code == 0
    assert result.exception is None
    assert calls == ["https://x/topic/1", "https://x/topic/2"]

    saved = read_posts(raw_path, RawPost)
    assert [p.id for p in saved] == ["t2"]


def test_scrape_continues_to_next_brand_when_one_brand_raises_fetcherror(tmp_path, monkeypatch):
    """Regression test for Finding A: a FetchError raised from discovery/
    pagination (discover_subforums or iter_topic_urls' own fetch() calls,
    once base.py's retries are exhausted) for one brand previously propagated
    all the way out of _scrape_automobile_propre, aborting the ENTIRE run and
    preventing the other 14 brands from ever being attempted. This asserts
    that "Renault" failing during discover_subforums does not prevent
    "Peugeot" -- discovered and iterated afterwards -- from being scraped and
    persisted."""
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    class MultiBrandScraper:
        def __init__(self, *args, **kwargs):
            pass

        def discover_brand_forums(self):
            return {
                "Renault": "https://x/forums/renault",
                "Peugeot": "https://x/forums/peugeot",
            }

        def discover_subforums(self, brand_url):
            if brand_url == "https://x/forums/renault":
                raise FetchError("échec après 3 tentatives sur la page de la marque")
            return {"208": "https://x/forums/peugeot/208"}

        def iter_topic_urls(self, subforum_url, scope):
            if subforum_url == "https://x/forums/peugeot/208":
                yield "https://x/topic/peugeot-1"
            return

        def scrape_topic(self, topic_url, brand, model):
            return [make_raw_post("p1", "la climatisation fait du bruit")]

    monkeypatch.setattr(main, "AutomobilePropreScraper", MultiBrandScraper)

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "automobile_propre"])

    assert result.exit_code == 0
    assert result.exception is None

    saved_ids = {p.id for p in read_posts(raw_path, RawPost)}
    assert "p1" in saved_ids


def test_scrape_source_all_still_runs_second_source_when_first_fails_entirely(
    tmp_path, monkeypatch
):
    """Regression test for Finding A: with --source all, a FetchError that
    escapes the whole automobile_propre run (e.g. its very first discovery
    call failing outright, before any per-brand guard even applies)
    previously propagated out of the scrape() command entirely, so caradisiac
    -- the second source -- was never even attempted. This asserts caradisiac
    still runs and its post is persisted despite automobile_propre's total
    failure."""
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    class TotallyFailingAutomobilePropreScraper:
        def __init__(self, *args, **kwargs):
            pass

        def discover_brand_forums(self):
            raise FetchError("échec après 3 tentatives sur la page des forums")

    def scrape_avis_fn(avis_url):
        return make_raw_post("c1", "la ventilation est bruyante")

    monkeypatch.setattr(
        main, "AutomobilePropreScraper", TotallyFailingAutomobilePropreScraper
    )
    monkeypatch.setattr(main, "CaradisiacScraper", make_fake_caradisiac_scraper(scrape_avis_fn))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "all"])

    assert result.exit_code == 0
    assert result.exception is None

    saved_ids = {p.id for p in read_posts(raw_path, RawPost)}
    assert "c1" in saved_ids


def test_scrape_topic_raising_unexpected_exception_continues_with_others(tmp_path, monkeypatch):
    """Regression test for Finding B: previously only FetchError was caught
    around the single-topic scrape_topic() call. Any other exception (e.g. an
    AttributeError from a JSON-LD parsing bug, or a pydantic ValidationError
    from a malformed datePublished) crashed the whole run -- and, since the
    checkpoint only records a page as done after ALL its topics are consumed,
    a rerun would re-fetch the same page, hit the same item, and crash again
    identically, forever. This asserts an unexpected (non-FetchError)
    exception on one topic is logged and skipped, and the next topic in the
    same brand/model is still scraped and persisted."""
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    calls = []

    def scrape_topic_fn(topic_url):
        calls.append(topic_url)
        if topic_url == "https://x/topic/1":
            raise AttributeError("'NoneType' object has no attribute 'get'")
        return [make_raw_post("t2", "la climatisation ne refroidit plus")]

    monkeypatch.setattr(
        main, "AutomobilePropreScraper", make_fake_automobile_propre_scraper(scrape_topic_fn)
    )

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "automobile_propre"])

    assert result.exit_code == 0
    assert result.exception is None
    assert calls == ["https://x/topic/1", "https://x/topic/2"]

    saved = read_posts(raw_path, RawPost)
    assert [p.id for p in saved] == ["t2"]


def make_fake_caradisiac_scraper(scrape_avis_fn):
    """Builds a stub class usable as a drop-in replacement for
    main.CaradisiacScraper, exposing two avis discovered under a single brand
    hub/model hub. `scrape_avis_fn(avis_url)` is invoked for each avis and its
    return value (or raised exception) becomes the result of `scrape_avis`.
    """

    class FakeCaradisiacScraper:
        def __init__(self, *args, **kwargs):
            pass

        def discover_brand_hubs(self):
            return {"renault": "https://x/auto--renault/"}

        def iter_model_hub_urls(self, brand_hub_url, scope):
            yield "https://x/modele--renault-zoe/avis/"

        def iter_avis_urls(self, model_hub_url, scope):
            yield "https://x/avis/1"
            yield "https://x/avis/2"

        def scrape_avis(self, avis_url):
            return scrape_avis_fn(avis_url)

    return FakeCaradisiacScraper


def test_scrape_caradisiac_skips_avis_raising_fetcherror_and_continues_with_others(
    tmp_path, monkeypatch
):
    """Same regression as the automobile_propre variant, for caradisiac's
    _scrape_caradisiac helper: a FetchError on one avis must not abort the
    rest of the run."""
    raw_path = tmp_path / "posts_raw.parquet"
    checkpoint_path = tmp_path / "progress.json"
    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "checkpoint", str(checkpoint_path))

    calls = []

    def scrape_avis_fn(avis_url):
        calls.append(avis_url)
        if avis_url == "https://x/avis/1":
            raise FetchError("échec après 3 tentatives")
        return make_raw_post("a2", "la ventilation est bruyante")

    monkeypatch.setattr(main, "CaradisiacScraper", make_fake_caradisiac_scraper(scrape_avis_fn))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["scrape", "--source", "caradisiac"])

    assert result.exit_code == 0
    assert result.exception is None
    assert calls == ["https://x/avis/1", "https://x/avis/2"]

    saved = read_posts(raw_path, RawPost)
    assert [p.id for p in saved] == ["a2"]


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


def test_classify_survives_mid_run_crash_by_flushing_incrementally(tmp_path, monkeypatch):
    """Regression test mirroring
    test_scrape_writes_posts_incrementally_surviving_mid_run_crash: classify()
    previously accumulated all ClassifiedPost results in memory and only
    called append_posts once at the very end (except on the
    OllamaUnavailableError path). An uncaught exception of any other type --
    here a plain RuntimeError, not one of the two handled exception types --
    used to discard everything already classified. This asserts the first
    N-1 posts' results are still persisted on disk despite the crash on the
    Nth."""
    raw_path = tmp_path / "posts_raw.parquet"
    classified_path = tmp_path / "posts_classified.parquet"
    posts = [
        make_raw_post("p1", "le chauffage ne fonctionne plus"),
        make_raw_post("p2", "la climatisation est en panne"),
        make_raw_post("p3", "buée persistante sur le pare-brise"),
    ]
    append_posts(raw_path, posts)

    monkeypatch.setitem(main.SETTINGS["paths"], "raw_posts", str(raw_path))
    monkeypatch.setitem(main.SETTINGS["paths"], "classified_posts", str(classified_path))

    calls = []

    def classify_fn(post):
        calls.append(post.id)
        if post.id == "p3":
            raise RuntimeError("crash inattendu pendant la classification")
        return make_llm_result()

    monkeypatch.setattr(main, "LLMClassifier", make_fake_llm_classifier(classify_fn))

    runner = CliRunner()
    result = runner.invoke(main.cli, ["classify"])

    assert calls == ["p1", "p2", "p3"]
    assert result.exception is not None

    saved = read_posts(classified_path, ClassifiedPost)
    assert {p.id for p in saved} == {"p1", "p2"}


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
