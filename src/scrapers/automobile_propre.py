from datetime import datetime, timezone
from typing import Iterator, Optional

from src.scrapers.base import BaseScraper, HTTPStatusError
from src.storage.models import RawPost, make_post_id
from src.utils.checkpoint import CheckpointStore

FORUMS_URL = "https://forums.automobile-propre.com/forums/"


class AutomobilePropreScraper(BaseScraper):
    def __init__(self, target_brands: list[str], checkpoint_store: CheckpointStore, **kwargs):
        super().__init__("automobile_propre", **kwargs)
        self.target_brands = {b.lower() for b in target_brands}
        self.checkpoint = checkpoint_store

    def discover_brand_forums(self) -> dict[str, str]:
        soup = self.fetch(FORUMS_URL)
        links = soup.select("[data-ips-hook='title'] a")
        forums = {}
        for link in links:
            heading = link.get_text(strip=True)
            if heading.lower() in self.target_brands:
                forums[heading] = link["href"]
            else:
                self.logger.info(
                    "Forum non ciblé ignoré: %r (aucune correspondance dans target_brands)",
                    heading,
                )
        return forums

    def discover_subforums(self, brand_url: str) -> dict[str, str]:
        soup = self.fetch(brand_url)
        links = soup.select("[data-ips-hook='title'] a")
        return {
            link.get_text(strip=True): link["href"]
            for link in links
            if "/forum/" in link["href"] and link["href"] != brand_url
        }

    def iter_topic_urls(self, subforum_url: str, scope: str) -> Iterator[str]:
        progress = self.checkpoint.get_progress(scope)
        seen_ids = set(progress["seen_ids"])
        page = progress["last_page"] + 1
        url = subforum_url if page == 1 else f"{subforum_url}?page={page}"

        while True:
            try:
                soup = self.fetch(url)
            except HTTPStatusError:
                break

            links = soup.select("[data-ips-hook='title'] a")
            topic_urls = [link["href"] for link in links if "/topic/" in link["href"]]
            new_urls = [u for u in topic_urls if u not in seen_ids]

            if not new_urls:
                break

            for topic_url in new_urls:
                seen_ids.add(topic_url)
                yield topic_url

            self.checkpoint.update_progress(scope, last_page=page, seen_ids=seen_ids)

            next_link = soup.select_one(
                "li.ipsPagination_next a, [data-role='tablePagination'] a[rel='next']"
            )
            if not next_link:
                break
            page += 1
            url = next_link["href"]

    def scrape_topic(self, topic_url: str, brand: str, model: Optional[str]) -> list[RawPost]:
        soup = self.fetch(topic_url)
        posts = []

        for article in soup.select("article[id^='elPost'], article[id^='elComment']"):
            body = article.select_one("[data-role='commentContent']")
            if not body:
                continue

            author_el = article.select_one(
                ".ipsEntry__username a, .ipsUsername, .ipsComment_author a[href*='/profile/']"
            )
            time_el = article.select_one("time[datetime]")

            text = body.get_text(strip=True)
            author = author_el.get_text(strip=True) if author_el else None
            published_at = time_el["datetime"] if time_el else None
            post_id = make_post_id("automobile_propre", topic_url, author or "", published_at or "")

            posts.append(
                RawPost(
                    id=post_id,
                    source="automobile_propre",
                    brand_source=brand,
                    model_source=model,
                    text=text,
                    author=author,
                    published_at=published_at,
                    url=topic_url,
                    scraped_at=datetime.now(timezone.utc),
                )
            )

        return posts
