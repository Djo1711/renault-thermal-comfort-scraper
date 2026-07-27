import responses

from src.scrapers.caradisiac import CaradisiacScraper
from src.utils.checkpoint import CheckpointStore
from tests.conftest import load_fixture


def make_scraper(tmp_path, target_brands=None):
    checkpoint = CheckpointStore(tmp_path / "progress.json")
    return CaradisiacScraper(
        target_brands=target_brands or ["Renault", "Peugeot"],
        checkpoint_store=checkpoint,
        min_delay=0,
        max_delay=0,
    )


@responses.activate
def test_discover_brand_hubs_filters_target_brands(tmp_path):
    responses.add(
        responses.GET,
        "https://www.caradisiac.com/avis/",
        body=load_fixture("caradisiac_avis_index.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    hubs = scraper.discover_brand_hubs()
    assert hubs == {
        "renault": "https://www.caradisiac.com/auto--renault/",
        "peugeot": "https://www.caradisiac.com/auto--peugeot/",
    }


@responses.activate
def test_iter_model_hub_urls_stops_when_page_repeats(tmp_path):
    # On the real site, model-hub hrefs found on a brand-hub avis-listing page
    # are RELATIVE (e.g. "/modele--renault-zoe/avis/"), not absolute -- the
    # fixture mirrors that.
    brand_hub = "https://www.caradisiac.com/auto--renault/"
    responses.add(
        responses.GET, f"{brand_hub}avis/", body=load_fixture("caradisiac_brand_hub_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        f"{brand_hub}avis_2/",
        body=load_fixture("caradisiac_brand_hub_page1.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    urls = list(scraper.iter_model_hub_urls(brand_hub, scope="caradisiac:renault"))
    # Even though the fixture's href attributes are relative, the scraper must
    # resolve them to absolute URLs before yielding them.
    assert urls == [
        "https://www.caradisiac.com/modele--renault-megane-e-tech/avis/",
        "https://www.caradisiac.com/modele--renault-zoe/avis/",
    ]
    assert all(u.startswith("https://www.caradisiac.com/") for u in urls)
    assert len(responses.calls) == 2


@responses.activate
def test_iter_avis_urls_stops_when_page_repeats(tmp_path):
    # model_hub is already the page-1 avis-listing URL (it ends in "avis/"),
    # exactly as produced by iter_model_hub_urls on the real site.
    model_hub = "https://www.caradisiac.com/modele--renault-megane-e-tech/avis/"
    responses.add(
        responses.GET, model_hub, body=load_fixture("caradisiac_model_hub_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        "https://www.caradisiac.com/modele--renault-megane-e-tech/avis_2/",
        body=load_fixture("caradisiac_model_hub_page2_repeat.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    urls = list(scraper.iter_avis_urls(model_hub, scope="caradisiac:renault:megane-e-tech"))
    # Fixture hrefs are relative; they must be resolved to absolute URLs.
    assert urls == [
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/",
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/def456/",
    ]
    assert len(responses.calls) == 2


@responses.activate
def test_iter_avis_urls_page_url_has_no_double_avis_segment(tmp_path):
    # Regression test for the "/avis/avis/" bug: model_hub_url already ends in
    # "avis/" (it's the page-1 listing URL, not a bare model root), so page 2
    # must be requested at ".../avis_2/", never ".../avis/avis_2/" or
    # ".../avis/avis/".
    model_hub = "https://www.caradisiac.com/modele--renault-megane-e-tech/avis/"
    responses.add(
        responses.GET, model_hub, body=load_fixture("caradisiac_model_hub_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        "https://www.caradisiac.com/modele--renault-megane-e-tech/avis_2/",
        body=load_fixture("caradisiac_model_hub_page2_repeat.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    list(scraper.iter_avis_urls(model_hub, scope="caradisiac:renault:megane-e-tech:dbl"))

    requested_urls = [call.request.url for call in responses.calls]
    assert requested_urls == [
        "https://www.caradisiac.com/modele--renault-megane-e-tech/avis/",
        "https://www.caradisiac.com/modele--renault-megane-e-tech/avis_2/",
    ]
    assert "avis/avis" not in requested_urls[1]


@responses.activate
def test_scrape_avis_extracts_from_json_ld_and_casts_year_to_str(tmp_path):
    avis_url = "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/"
    responses.add(responses.GET, avis_url, body=load_fixture("caradisiac_avis_page.html"), status=200)
    scraper = make_scraper(tmp_path)
    post = scraper.scrape_avis(avis_url)
    assert post is not None
    assert post.brand_source == "Renault"
    assert post.model_source == "Megane E-Tech"
    assert post.year_source == "2022"
    assert post.url == avis_url
    assert "désembuer" in post.text
    assert "Confort de conduite" in post.text
    assert "Désembuage lent" in post.text


@responses.activate
def test_scrape_avis_returns_none_for_non_target_brand(tmp_path):
    avis_url = "https://www.caradisiac.com/avis--modele--tesla-model3/xyz/"
    responses.add(
        responses.GET,
        avis_url,
        body=load_fixture("caradisiac_avis_page.html").replace("Renault", "Tesla"),
        status=200,
    )
    scraper = make_scraper(tmp_path, target_brands=["Renault"])
    post = scraper.scrape_avis(avis_url)
    assert post is None


@responses.activate
def test_iter_avis_urls_resumes_from_checkpoint(tmp_path):
    model_hub = "https://www.caradisiac.com/modele--renault-megane-e-tech/avis/"
    scope = "caradisiac:renault:megane-e-tech"
    seen_urls = {
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/",
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/def456/",
    }
    checkpoint = CheckpointStore(tmp_path / "progress.json")
    checkpoint.update_progress(scope, last_page=1, seen_ids=seen_urls)

    responses.add(
        responses.GET,
        "https://www.caradisiac.com/modele--renault-megane-e-tech/avis_2/",
        body=load_fixture("caradisiac_model_hub_page2_repeat.html"),
        status=200,
    )

    scraper = CaradisiacScraper(
        target_brands=["Renault", "Peugeot"],
        checkpoint_store=checkpoint,
        min_delay=0,
        max_delay=0,
    )
    urls = list(scraper.iter_avis_urls(model_hub, scope=scope))

    # Resume must jump straight to page 2 (no request to the page-1/avis/ url)
    # and yield nothing new since page 2 repeats already-seen avis links.
    assert urls == []
    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "https://www.caradisiac.com/modele--renault-megane-e-tech/avis_2/"

    progress = checkpoint.get_progress(scope)
    assert progress["last_page"] == 1
    assert set(progress["seen_ids"]) == seen_urls
