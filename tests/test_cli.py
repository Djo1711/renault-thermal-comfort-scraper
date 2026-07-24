from datetime import datetime, timezone

from click.testing import CliRunner

import main
from src.storage.models import ClassifiedPost
from src.storage.writer import append_posts


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
