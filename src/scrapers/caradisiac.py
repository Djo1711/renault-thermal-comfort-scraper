import json
import unicodedata
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional
from urllib.parse import urljoin

from src.scrapers.base import BaseScraper, HTTPStatusError
from src.storage.models import RawPost, make_post_id
from src.utils.checkpoint import CheckpointStore

BASE_URL = "https://www.caradisiac.com"
AVIS_INDEX_URL = f"{BASE_URL}/avis/"


def _normalize(s: str) -> str:
    """Strip diacritics and lowercase, so "Citroën" (config/vehicle_list.yaml,
    with the diaeresis) matches the ASCII URL slug "citroen" (from
    "/auto--citroen/") instead of silently never matching."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()


class CaradisiacScraper(BaseScraper):
    def __init__(self, target_brands: list[str], checkpoint_store: CheckpointStore, **kwargs):
        super().__init__("caradisiac", **kwargs)
        self.target_brands = {_normalize(b) for b in target_brands}
        self.checkpoint = checkpoint_store

    def discover_brand_hubs(self) -> dict[str, str]:
        soup = self.fetch(AVIS_INDEX_URL)
        hubs = {}
        for link in soup.select("a[href*='/auto--']"):
            href = urljoin(BASE_URL, link["href"])
            slug = href.strip("/").split("--")[-1]
            normalized_slug = _normalize(slug)
            if normalized_slug in self.target_brands:
                hubs[normalized_slug] = href
            else:
                self.logger.info(
                    "Marque non ciblée ignorée sur /auto--%s/ (aucune correspondance dans "
                    "target_brands)",
                    slug,
                )
        return hubs

    def _iter_paginated_links(
        self, page_url_fn: Callable[[int], str], link_selector: str, scope: str
    ) -> Iterator[str]:
        progress = self.checkpoint.get_progress(scope)
        seen = set(progress["seen_ids"])
        page = progress["last_page"] + 1

        while True:
            url = page_url_fn(page)
            try:
                soup = self.fetch(url)
            except HTTPStatusError:
                break

            links = [urljoin(BASE_URL, a["href"]) for a in soup.select(link_selector)]
            new_links = [u for u in links if u not in seen]

            if not new_links:
                break

            for link in new_links:
                seen.add(link)
                yield link

            self.checkpoint.update_progress(scope, last_page=page, seen_ids=seen)
            page += 1

    def iter_model_hub_urls(self, brand_hub_url: str, scope: str) -> Iterator[str]:
        def page_url(page: int) -> str:
            return f"{brand_hub_url}avis/" if page == 1 else f"{brand_hub_url}avis_{page}/"

        yield from self._iter_paginated_links(page_url, "a[href*='/modele--']", scope)

    def iter_avis_urls(self, model_hub_url: str, scope: str) -> Iterator[str]:
        # model_hub_url is already the page-1 avis-listing URL (it ends in
        # "avis/", as yielded by iter_model_hub_urls) -- it is NOT a bare
        # model root that we should append "avis/" to. Doing so produces a
        # double "avis/avis/" segment. Strip the trailing "avis/" once to
        # recover the model's base prefix, then rebuild each page URL from it.
        suffix = "avis/"
        prefix = model_hub_url[: -len(suffix)] if model_hub_url.endswith(suffix) else model_hub_url

        def page_url(page: int) -> str:
            return f"{prefix}avis/" if page == 1 else f"{prefix}avis_{page}/"

        yield from self._iter_paginated_links(page_url, "a[href*='/avis--modele--']", scope)

    def scrape_avis(self, avis_url: str) -> Optional[RawPost]:
        soup = self.fetch(avis_url)
        data = None
        for script in soup.select("script[type='application/ld+json']"):
            if not script.string:
                continue
            try:
                candidate = json.loads(script.string)
            except (json.JSONDecodeError, TypeError):
                # Malformed JSON-LD block on the page -- skip it and keep
                # checking the remaining scripts rather than crashing.
                continue
            if candidate.get("@type") == "Review":
                data = candidate
                break

        if data is None:
            self.logger.warning(
                "Aucun bloc JSON-LD de type Review trouvé sur %s", avis_url
            )
            return None

        item = data.get("itemReviewed", {})
        brand = item.get("brand", {}).get("name")
        if not brand or _normalize(brand) not in self.target_brands:
            return None

        model = item.get("model")
        year_raw = item.get("vehicleModelDate")
        year_source = str(year_raw) if year_raw is not None else None

        text = data.get("reviewBody", "")
        plus_items = [li.get_text(strip=True) for li in soup.select("div.blocLesPlus li")]
        minus_items = [li.get_text(strip=True) for li in soup.select("div.blocLesMoins li")]
        if plus_items:
            text += "\nPoints positifs: " + "; ".join(plus_items)
        if minus_items:
            text += "\nPoints négatifs: " + "; ".join(minus_items)

        author = data.get("author", {}).get("name")
        published_at = data.get("datePublished")
        post_id = make_post_id("caradisiac", avis_url, author or "", published_at or "")

        return RawPost(
            id=post_id,
            source="caradisiac",
            brand_source=brand,
            model_source=model,
            year_source=year_source,
            text=text,
            author=author,
            published_at=published_at,
            url=avis_url,
            scraped_at=datetime.now(timezone.utc),
        )
