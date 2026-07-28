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
def test_discover_brand_hubs_matches_citroen_despite_diacritic_mismatch(tmp_path):
    # Regression test: target_brands comes from config/vehicle_list.yaml as
    # "Citroën" (with the diaeresis), but the site's URL slug is the ASCII
    # "citroen" (from "/auto--citroen/"). A plain .lower() comparison never
    # matches these, so Citroën's real hub was silently never discovered even
    # though it's one of the 15 target brands. This test FAILS against the
    # pre-fix code (a plain .lower() comparison) and passes once both sides
    # are diacritic-normalized before comparison.
    responses.add(
        responses.GET,
        "https://www.caradisiac.com/avis/",
        body='<html><body><a href="/auto--citroen/">Citroën</a></body></html>',
        status=200,
    )
    scraper = make_scraper(tmp_path, target_brands=["Citroën"])
    hubs = scraper.discover_brand_hubs()
    assert hubs == {"citroen": "https://www.caradisiac.com/auto--citroen/"}


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
def test_scrape_avis_finds_review_block_when_not_the_first_json_ld_script(tmp_path):
    # Regression test: scrape_avis used select_one(), returning only the FIRST
    # script[type='application/ld+json'] block. Real caradisiac pages often
    # have multiple JSON-LD blocks (BreadcrumbList, Organization, etc.) in
    # addition to the Review block -- if Review isn't first, the avis was
    # silently and permanently dropped (the checkpoint already marks the URL
    # as seen). This test FAILS against select_one() and passes once all
    # blocks are checked for @type == "Review".
    avis_url = "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/multi-ld/"
    page = """
    <html><body>
    <script type="application/ld+json">
    {"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": []}
    </script>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Review",
      "reviewBody": "Bonne isolation mais chauffage lent en hiver.",
      "datePublished": "2023-11-02",
      "itemReviewed": {
        "brand": {"name": "Renault"},
        "model": "Megane E-Tech",
        "vehicleModelDate": 2022
      },
      "author": {"name": "Marc D."}
    }
    </script>
    </body></html>
    """
    responses.add(responses.GET, avis_url, body=page, status=200)
    scraper = make_scraper(tmp_path)
    post = scraper.scrape_avis(avis_url)
    assert post is not None
    assert post.brand_source == "Renault"
    assert "chauffage lent" in post.text


@responses.activate
def test_scrape_avis_skips_malformed_json_ld_block_and_finds_valid_review(tmp_path):
    # A malformed (invalid JSON) script block must not crash scrape_avis --
    # it should be skipped, continuing to check the next block for a valid
    # Review.
    avis_url = "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/malformed-ld/"
    page = """
    <html><body>
    <script type="application/ld+json">
    { this is not valid json at all
    </script>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Review",
      "reviewBody": "Bonne voiture, buée rapide sur les vitres.",
      "datePublished": "2023-11-02",
      "itemReviewed": {
        "brand": {"name": "Renault"},
        "model": "Megane E-Tech",
        "vehicleModelDate": 2022
      },
      "author": {"name": "Marc D."}
    }
    </script>
    </body></html>
    """
    responses.add(responses.GET, avis_url, body=page, status=200)
    scraper = make_scraper(tmp_path)
    post = scraper.scrape_avis(avis_url)
    assert post is not None
    assert "buée rapide" in post.text


@responses.activate
def test_scrape_avis_handles_brand_as_plain_string_shorthand(tmp_path):
    # Regression test for Finding B: schema.org's "brand" property is valid
    # JSON-LD either as a Brand object ({"name": "Renault"}) or as a plain
    # string shorthand ("Renault") -- item.get("brand", {}).get("name")
    # crashes with AttributeError: 'str' object has no attribute 'get' on
    # the string form. This test FAILS against that code and passes once
    # the brand field is normalized to a name string in either shape.
    avis_url = "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/brand-string/"
    page = """
    <html><body>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Review",
      "reviewBody": "Chauffage capricieux en hiver.",
      "datePublished": "2023-11-02",
      "itemReviewed": {
        "brand": "Renault",
        "model": "Megane E-Tech",
        "vehicleModelDate": 2022
      },
      "author": {"name": "Marc D."}
    }
    </script>
    </body></html>
    """
    responses.add(responses.GET, avis_url, body=page, status=200)
    scraper = make_scraper(tmp_path)
    post = scraper.scrape_avis(avis_url)
    assert post is not None
    assert post.brand_source == "Renault"


@responses.activate
def test_scrape_avis_skips_top_level_json_ld_array_block(tmp_path):
    # Regression test for Finding B: a JSON-LD <script> block can legally
    # contain a top-level array (e.g. multiple structured-data items bundled
    # together) rather than an object -- candidate.get("@type") crashes with
    # AttributeError: 'list' object has no attribute 'get' on that block.
    # This test FAILS against that code and passes once non-dict blocks are
    # skipped, still finding the later, valid Review block.
    avis_url = "https://www.caradisiac.com/avis--modele--renault-megane-e-tech/array-ld/"
    page = """
    <html><body>
    <script type="application/ld+json">
    [{"@context": "https://schema.org", "@type": "BreadcrumbList", "itemListElement": []}]
    </script>
    <script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "Review",
      "reviewBody": "Ventilation bruyante mais efficace.",
      "datePublished": "2023-11-02",
      "itemReviewed": {
        "brand": {"name": "Renault"},
        "model": "Megane E-Tech",
        "vehicleModelDate": 2022
      },
      "author": {"name": "Marc D."}
    }
    </script>
    </body></html>
    """
    responses.add(responses.GET, avis_url, body=page, status=200)
    scraper = make_scraper(tmp_path)
    post = scraper.scrape_avis(avis_url)
    assert post is not None
    assert "Ventilation bruyante" in post.text


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
