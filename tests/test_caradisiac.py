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
    assert urls == [
        "https://www.caradisiac.com/modele--renault-megane-e-tech/",
        "https://www.caradisiac.com/modele--renault-zoe/",
    ]
    assert len(responses.calls) == 2


@responses.activate
def test_iter_avis_urls_stops_when_page_repeats(tmp_path):
    model_hub = "https://www.caradisiac.com/modele--renault-megane-e-tech/"
    responses.add(
        responses.GET, f"{model_hub}avis/", body=load_fixture("caradisiac_model_hub_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        f"{model_hub}avis_2/",
        body=load_fixture("caradisiac_model_hub_page2_repeat.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    urls = list(scraper.iter_avis_urls(model_hub, scope="caradisiac:renault:megane-e-tech"))
    assert urls == [
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/",
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/def456/",
    ]
    assert len(responses.calls) == 2


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
    model_hub = "https://www.caradisiac.com/modele--renault-megane-e-tech/"
    scope = "caradisiac:renault:megane-e-tech"
    seen_urls = {
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/abc123/",
        "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/def456/",
    }
    checkpoint = CheckpointStore(tmp_path / "progress.json")
    checkpoint.update_progress(scope, last_page=1, seen_ids=seen_urls)

    responses.add(
        responses.GET,
        f"{model_hub}avis_2/",
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
    assert responses.calls[0].request.url == f"{model_hub}avis_2/"

    progress = checkpoint.get_progress(scope)
    assert progress["last_page"] == 1
    assert set(progress["seen_ids"]) == seen_urls
