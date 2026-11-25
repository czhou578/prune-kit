import os

import pytest

from prune_kit.resolve import ModelNotFound, hubs, resolve


def make_repo(hub, name, snapshots=("aaa",)):
    for i, snap in enumerate(snapshots):
        folder = hub / ("models--" + name.replace("/", "--")) / "snapshots" / snap
        folder.mkdir(parents=True)
        os.utime(folder, (1000 + i, 1000 + i))
    return hub / ("models--" + name.replace("/", "--"))


@pytest.fixture
def hub(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "none"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "hub").mkdir()
    return tmp_path / "hub"


def test_a_plain_folder(tmp_path):
    assert resolve(str(tmp_path)) == tmp_path


def test_full_name_or_repo_folder_gives_the_newest_snapshot(hub):
    repo = make_repo(hub, "org/Model-A", snapshots=("old", "new"))
    assert resolve("org/Model-A") == resolve(str(repo)) == repo / "snapshots" / "new"


def test_a_part_of_a_name(hub):
    make_repo(hub, "org/Model-A")
    make_repo(hub, "org/Other")
    assert resolve("model-a").parent.parent.name == "models--org--Model-A"


def test_an_exact_name_wins_over_a_part(hub):
    make_repo(hub, "org/Model")
    make_repo(hub, "org/Model-Fast")
    assert resolve("org/Model").parent.parent.name == "models--org--Model"


def test_more_than_one_match(hub):
    make_repo(hub, "org/Model-A")
    make_repo(hub, "org/Model-B")
    with pytest.raises(ModelNotFound, match="org/Model-A, org/Model-B"):
        resolve("model")


def test_no_match_and_no_snapshot(hub):
    with pytest.raises(ModelNotFound, match="no folder"):
        resolve("org/Missing")
    (hub / "models--org--Empty").mkdir()
    with pytest.raises(ModelNotFound, match="no snapshot"):
        resolve("org/Empty")


def test_extra_hub_comes_first(hub, tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    assert hubs([str(extra)]) == [extra, hub]
