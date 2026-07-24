from datetime import datetime, timezone

from src.storage.models import RawPost, make_post_id
from src.storage.writer import append_posts, read_existing_ids, read_posts


def make_post(post_id_suffix: str) -> RawPost:
    return RawPost(
        id=make_post_id("automobile_propre", f"https://x/topic/{post_id_suffix}", "auteur", "2024-01-01"),
        source="automobile_propre",
        brand_source="Renault",
        text=f"texte {post_id_suffix}",
        author="auteur",
        published_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        url=f"https://x/topic/{post_id_suffix}",
        scraped_at=datetime.now(timezone.utc),
    )


def test_append_posts_writes_new_file(tmp_path):
    path = tmp_path / "posts_raw.parquet"
    written = append_posts(path, [make_post("1"), make_post("2")])
    assert written == 2
    assert path.exists()
    assert read_existing_ids(path) == {make_post("1").id, make_post("2").id}


def test_append_posts_skips_already_present_ids(tmp_path):
    path = tmp_path / "posts_raw.parquet"
    append_posts(path, [make_post("1")])
    written = append_posts(path, [make_post("1"), make_post("2")])
    assert written == 1
    assert len(read_posts(path, RawPost)) == 2


def test_read_existing_ids_empty_when_no_file(tmp_path):
    assert read_existing_ids(tmp_path / "missing.parquet") == set()


def test_read_posts_reconstructs_pydantic_models(tmp_path):
    path = tmp_path / "posts_raw.parquet"
    append_posts(path, [make_post("1")])
    posts = read_posts(path, RawPost)
    assert len(posts) == 1
    assert isinstance(posts[0], RawPost)
    assert posts[0].id == make_post("1").id
    assert posts[0].brand_source == "Renault"


def test_read_posts_empty_list_when_no_file(tmp_path):
    assert read_posts(tmp_path / "missing.parquet", RawPost) == []


def test_append_posts_empty_list_nonexistent_file(tmp_path):
    """Appending empty list to nonexistent file should be a no-op, returning 0."""
    path = tmp_path / "posts_raw.parquet"
    written = append_posts(path, [])
    assert written == 0
    # File should not exist or be usable
    # Verify that a subsequent call with real posts works correctly
    written_next = append_posts(path, [make_post("1")])
    assert written_next == 1
    assert read_existing_ids(path) == {make_post("1").id}


def test_append_posts_empty_list_existing_file(tmp_path):
    """Appending empty list to existing file should be a no-op, preserving content."""
    path = tmp_path / "posts_raw.parquet"
    # First, populate the file
    append_posts(path, [make_post("1"), make_post("2")])
    original_ids = read_existing_ids(path)
    original_posts = read_posts(path, RawPost)
    assert len(original_posts) == 2

    # Now try to append an empty list
    written = append_posts(path, [])
    assert written == 0

    # Verify content is unchanged
    assert read_existing_ids(path) == original_ids
    assert read_posts(path, RawPost) == original_posts
