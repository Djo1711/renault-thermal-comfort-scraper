import responses

from src.scrapers.automobile_propre import AutomobilePropreScraper
from src.utils.checkpoint import CheckpointStore
from tests.conftest import load_fixture


def make_scraper(tmp_path, target_brands=None):
    checkpoint = CheckpointStore(tmp_path / "progress.json")
    return AutomobilePropreScraper(
        target_brands=target_brands or ["Renault", "Peugeot"],
        checkpoint_store=checkpoint,
        min_delay=0,
        max_delay=0,
    )


@responses.activate
def test_discover_brand_forums_filters_target_brands(tmp_path):
    responses.add(
        responses.GET,
        "https://forums.automobile-propre.com/forums/",
        body=load_fixture("automobile_propre_forums_list.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    brands = scraper.discover_brand_forums()
    assert brands == {
        "Renault": "https://forums.automobile-propre.com/forums/forum/12-renault/",
        "Peugeot": "https://forums.automobile-propre.com/forums/forum/56-peugeot/",
    }


@responses.activate
def test_discover_subforums_returns_model_links(tmp_path):
    responses.add(
        responses.GET,
        "https://forums.automobile-propre.com/forums/forum/12-renault/",
        body=load_fixture("automobile_propre_brand_page.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    subforums = scraper.discover_subforums("https://forums.automobile-propre.com/forums/forum/12-renault/")
    assert subforums == {
        "Megane E-Tech": "https://forums.automobile-propre.com/forums/forum/45-megane-e-tech/",
        "Zoe": "https://forums.automobile-propre.com/forums/forum/46-zoe/",
    }


@responses.activate
def test_iter_topic_urls_stops_when_page_repeats_same_topics(tmp_path):
    subforum_url = "https://forums.automobile-propre.com/forums/forum/45-megane-e-tech/"
    responses.add(
        responses.GET, subforum_url, body=load_fixture("automobile_propre_topic_list_page1.html"), status=200
    )
    responses.add(
        responses.GET,
        f"{subforum_url}?page=2",
        body=load_fixture("automobile_propre_topic_list_page2_repeat.html"),
        status=200,
    )
    scraper = make_scraper(tmp_path)
    urls = list(scraper.iter_topic_urls(subforum_url, scope="automobile_propre:renault:megane-e-tech"))
    assert urls == [
        "https://forums.automobile-propre.com/forums/topic/1001-buee-persistante/",
        "https://forums.automobile-propre.com/forums/topic/1002-autonomie-hiver/",
    ]
    assert len(responses.calls) == 2


@responses.activate
def test_scrape_topic_extracts_posts(tmp_path):
    topic_url = "https://forums.automobile-propre.com/forums/topic/1001-buee-persistante/"
    responses.add(responses.GET, topic_url, body=load_fixture("automobile_propre_topic_page.html"), status=200)
    scraper = make_scraper(tmp_path)
    posts = scraper.scrape_topic(topic_url, brand="Renault", model="Megane E-Tech")
    assert len(posts) == 2
    assert "buée" in posts[0].text
    assert posts[0].author == "jdupont"
    assert posts[0].brand_source == "Renault"
    assert posts[0].model_source == "Megane E-Tech"
