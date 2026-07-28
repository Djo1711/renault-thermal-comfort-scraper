import responses

from src.scrapers.base import BaseScraper, FetchError, HTTPStatusError


@responses.activate
def test_fetch_sleeps_with_exponential_backoff_between_retries(monkeypatch):
    responses.add(responses.GET, "https://example.test/page", status=500)
    responses.add(responses.GET, "https://example.test/page", status=500)
    responses.add(responses.GET, "https://example.test/page", status=500)
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0, max_retries=3)

    sleep_calls = []
    monkeypatch.setattr(
        "src.scrapers.base.time.sleep", lambda seconds: sleep_calls.append(seconds)
    )

    try:
        scraper.fetch("https://example.test/page")
        assert False, "aurait dû lever FetchError"
    except FetchError:
        pass

    # 3 attempts -> 2 delays between them (no delay needed after the last
    # failed attempt since we're giving up), each strictly positive and
    # non-decreasing (exponential backoff).
    assert len(sleep_calls) == 2
    assert all(s > 0 for s in sleep_calls)
    assert sleep_calls[1] >= sleep_calls[0]


@responses.activate
def test_fetch_returns_parsed_soup_on_success():
    responses.add(
        responses.GET,
        "https://example.test/page",
        body="<html><body><p>ok</p></body></html>",
        status=200,
    )
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0)
    soup = scraper.fetch("https://example.test/page")
    assert soup.find("p").text == "ok"


@responses.activate
def test_fetch_raises_http_status_error_on_400_without_retry():
    responses.add(responses.GET, "https://example.test/page", status=400)
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0, max_retries=3)

    try:
        scraper.fetch("https://example.test/page")
        assert False, "aurait dû lever HTTPStatusError"
    except HTTPStatusError as exc:
        assert exc.status_code == 400

    assert len(responses.calls) == 1


@responses.activate
def test_fetch_retries_on_500_then_succeeds():
    responses.add(responses.GET, "https://example.test/page", status=500)
    responses.add(
        responses.GET,
        "https://example.test/page",
        body="<html><body><p>ok</p></body></html>",
        status=200,
    )
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0, max_retries=3)
    soup = scraper.fetch("https://example.test/page")
    assert soup.find("p").text == "ok"
    assert len(responses.calls) == 2


@responses.activate
def test_fetch_raises_after_max_retries_on_persistent_500():
    responses.add(responses.GET, "https://example.test/page", status=500)
    responses.add(responses.GET, "https://example.test/page", status=500)
    scraper = BaseScraper("test_source", min_delay=0, max_delay=0, max_retries=2)

    try:
        scraper.fetch("https://example.test/page")
        assert False, "aurait dû lever FetchError"
    except FetchError:
        pass

    assert len(responses.calls) == 2
