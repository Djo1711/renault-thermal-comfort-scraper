import sys

import click
from rich.console import Console
from rich.table import Table

from src.pipeline.export import export_all
from src.pipeline.llm_classifier import (
    LLMClassifier,
    LLMResponseInvalidError,
    OllamaUnavailableError,
)
from src.pipeline.prefilter import KeywordPrefilter
from src.scrapers.automobile_propre import AutomobilePropreScraper
from src.scrapers.base import FetchError
from src.scrapers.caradisiac import CaradisiacScraper
from src.storage.models import ClassifiedPost, RawPost
from src.storage.writer import append_posts, read_existing_ids, read_posts
from src.utils.checkpoint import CheckpointStore
from src.utils.config import load_yaml
from src.utils.logger import get_logger

console = Console()
SETTINGS = load_yaml("config/settings.yaml")
VEHICLE_LIST = load_yaml("config/vehicle_list.yaml")["brands"]


@click.group()
def cli():
    pass


def _scrape_automobile_propre(
    scraper: AutomobilePropreScraper, raw_posts_path: str, logger
) -> int:
    """Scrape all target brands/subforums/topics, writing each topic's posts to disk
    as soon as it is scraped so a mid-run interruption cannot lose already-scraped
    content. A single topic that fails after exhausting base.py's retries is
    logged and skipped rather than aborting the whole run. Returns the total
    number of newly written posts."""
    written_total = 0
    for brand, brand_url in scraper.discover_brand_forums().items():
        for model, model_url in scraper.discover_subforums(brand_url).items():
            scope = f"automobile_propre:{brand}:{model}"
            for topic_url in scraper.iter_topic_urls(model_url, scope):
                try:
                    topic_posts = scraper.scrape_topic(topic_url, brand=brand, model=model)
                except FetchError as exc:
                    logger.warning("Topic ignoré après échec de récupération %s: %s", topic_url, exc)
                    continue
                written_total += append_posts(raw_posts_path, topic_posts)

        # Topics can also be posted directly at the brand-forum level (no
        # model sub-forum) -- scrape the brand's own forum URL too, under a
        # distinct scope so it doesn't collide with any sub-forum's checkpoint.
        direct_scope = f"automobile_propre:{brand}:_direct"
        for topic_url in scraper.iter_topic_urls(brand_url, direct_scope):
            try:
                topic_posts = scraper.scrape_topic(topic_url, brand=brand, model=None)
            except FetchError as exc:
                logger.warning("Topic ignoré après échec de récupération %s: %s", topic_url, exc)
                continue
            written_total += append_posts(raw_posts_path, topic_posts)
    return written_total


def _scrape_caradisiac(scraper: CaradisiacScraper, raw_posts_path: str, logger) -> int:
    """Scrape all target brand hubs/model hubs/avis, writing each avis to disk as
    soon as it is scraped so a mid-run interruption cannot lose already-scraped
    content. A single avis that fails after exhausting base.py's retries is
    logged and skipped rather than aborting the whole run. Returns the total
    number of newly written posts."""
    written_total = 0
    for brand, brand_hub_url in scraper.discover_brand_hubs().items():
        model_scope = f"caradisiac:{brand}"
        for model_hub_url in scraper.iter_model_hub_urls(brand_hub_url, scope=model_scope):
            avis_scope = f"caradisiac:{brand}:{model_hub_url}"
            for avis_url in scraper.iter_avis_urls(model_hub_url, scope=avis_scope):
                try:
                    post = scraper.scrape_avis(avis_url)
                except FetchError as exc:
                    logger.warning("Avis ignoré après échec de récupération %s: %s", avis_url, exc)
                    continue
                if post:
                    written_total += append_posts(raw_posts_path, [post])
    return written_total


@cli.command()
@click.option(
    "--source", type=click.Choice(["automobile_propre", "caradisiac", "all"]), default="all"
)
def scrape(source):
    logger = get_logger("scraper", log_file="data/scraper.log")
    checkpoint = CheckpointStore(SETTINGS["paths"]["checkpoint"])
    sources = ["automobile_propre", "caradisiac"] if source == "all" else [source]

    common_kwargs = dict(
        checkpoint_store=checkpoint,
        min_delay=SETTINGS["rate_limit"]["min_seconds"],
        max_delay=SETTINGS["rate_limit"]["max_seconds"],
        timeout=SETTINGS["http"]["timeout_seconds"],
        max_retries=SETTINGS["http"]["max_retries"],
        user_agent=SETTINGS["http"]["user_agent"],
    )

    for src_name in sources:
        raw_posts_path = SETTINGS["paths"]["raw_posts"]
        if src_name == "automobile_propre":
            scraper = AutomobilePropreScraper(target_brands=VEHICLE_LIST, **common_kwargs)
            written = _scrape_automobile_propre(scraper, raw_posts_path, logger)
        else:
            scraper = CaradisiacScraper(target_brands=VEHICLE_LIST, **common_kwargs)
            written = _scrape_caradisiac(scraper, raw_posts_path, logger)

        logger.info("%s: %d nouveaux posts écrits", src_name, written)
        console.print(f"[green]{src_name}[/green] : {written} nouveaux posts")


FLUSH_EVERY = 10


@cli.command()
def classify():
    logger = get_logger("scraper", log_file="data/scraper.log")
    prefilter = KeywordPrefilter("config/keywords.yaml")
    llm = LLMClassifier(
        base_url=SETTINGS["ollama"]["base_url"],
        model=SETTINGS["ollama"]["model"],
        max_retries=SETTINGS["ollama"]["max_retries"],
    )

    raw_posts = read_posts(SETTINGS["paths"]["raw_posts"], RawPost)
    already_classified = read_existing_ids(SETTINGS["paths"]["classified_posts"])
    to_classify = [p for p in raw_posts if p.id not in already_classified]

    console.print(f"{len(to_classify)} posts à classifier ({len(raw_posts) - len(to_classify)} déjà faits)")

    # `results` is flushed to disk periodically (not just once at the very
    # end) so a crash mid-loop -- a KeyboardInterrupt or any other uncaught
    # exception, not only OllamaUnavailableError -- cannot discard already-
    # classified posts. append_posts is idempotent by id, so partial/repeated
    # flushes are safe.
    classified_path = SETTINGS["paths"]["classified_posts"]
    results: list[ClassifiedPost] = []
    written_total = 0

    def flush() -> None:
        nonlocal results, written_total
        written_total += append_posts(classified_path, results)
        results = []

    try:
        for post in to_classify:
            if not prefilter.is_relevant(post.text):
                continue
            try:
                llm_result = llm.classify(post)
                results.append(ClassifiedPost.from_raw_and_llm(post, llm_result))
            except OllamaUnavailableError:
                flush()
                console.print(
                    f"[red]Ollama injoignable — lance `ollama serve`. "
                    f"{written_total} posts classifiés avant l'arrêt.[/red]"
                )
                sys.exit(1)
            except LLMResponseInvalidError as exc:
                logger.warning("Classification échouée pour %s: %s", post.id, exc)
                results.append(ClassifiedPost.failed(post))

            if len(results) >= FLUSH_EVERY:
                flush()
    finally:
        flush()

    console.print(f"[green]{written_total}[/green] posts classifiés écrits")


@cli.command()
def export():
    outputs = export_all(SETTINGS["paths"]["classified_posts"], SETTINGS["paths"]["output_dir"])
    table = Table(title="Export terminé")
    table.add_column("Fichier")
    table.add_column("Chemin")
    for name, path in outputs.items():
        table.add_row(name, str(path))
    console.print(table)


if __name__ == "__main__":
    cli()
