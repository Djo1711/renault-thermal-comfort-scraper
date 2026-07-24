from src.utils.checkpoint import CheckpointStore


def test_new_checkpoint_returns_empty_progress(tmp_path):
    store = CheckpointStore(tmp_path / "progress.json")
    progress = store.get_progress("automobile_propre:renault:megane-e-tech")
    assert progress == {"last_page": 0, "seen_ids": []}


def test_update_progress_persists_to_disk_and_is_reloadable(tmp_path):
    checkpoint_path = tmp_path / "progress.json"
    store = CheckpointStore(checkpoint_path)
    store.update_progress(
        "automobile_propre:renault:megane-e-tech", last_page=2, seen_ids={"id1", "id2"}
    )

    reloaded = CheckpointStore(checkpoint_path)
    progress = reloaded.get_progress("automobile_propre:renault:megane-e-tech")
    assert progress["last_page"] == 2
    assert sorted(progress["seen_ids"]) == ["id1", "id2"]


def test_different_scopes_are_independent(tmp_path):
    store = CheckpointStore(tmp_path / "progress.json")
    store.update_progress("scope_a", last_page=1, seen_ids={"a"})
    store.update_progress("scope_b", last_page=3, seen_ids={"b"})

    assert store.get_progress("scope_a")["last_page"] == 1
    assert store.get_progress("scope_b")["last_page"] == 3
