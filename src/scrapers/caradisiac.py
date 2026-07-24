import json
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

from src.scrapers.base import BaseScraper, HTTPStatusError
from src.storage.models import RawPost, make_post_id
from src.utils.checkpoint import CheckpointStore

AVIS_INDEX_URL = "https://www.caradisiac.com/avis/"


class CaradisiacScraper(BaseScraper):
    def __init__(self, target_brands: list[str], checkpoint_store: CheckpointStore, **kwargs):
        super().__init__("caradisiac", **kwargs)
        self.target_brands = {b.lower() for b in target_brands}
        self.checkpoint = checkpoint_store

    def discover_brand_hubs(self) -> dict[str, str]:
        soup = self.fetch(AVIS_INDEX_URL)
        hubs = {}
        for link in soup.select("a[href*='/auto--']"):
            href = link["href"]
            slug = href.strip("/").split("--")[-1]
            if slug.lower() in self.target_brands:
                hubs[slug.lower()] = href
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

            links = [a["href"] for a in soup.select(link_selector)]
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
        def page_url(page: int) -> str:
            return f"{model_hub_url}avis/" if page == 1 else f"{model_hub_url}avis_{page}/"

        yield from self._iter_paginated_links(page_url, "a[href*='/avis--modele--']", scope)

    def scrape_avis(self, avis_url: str) -> Optional[RawPost]:
        soup = self.fetch(avis_url)
        script = soup.select_one("script[type='application/ld+json']")
        if not script or not script.string:
            return None

        data = json.loads(script.string)
        if data.get("@type") != "Review":
            return None

        item = data.get("itemReviewed", {})
        brand = item.get("brand", {}).get("name")
        if not brand or brand.lower() not in self.target_brands:
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
