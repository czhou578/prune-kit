import pytest

from prune_moe_kit import __version__, cli, stages


def test_stages_lists_each_stage_once(capsys):
    assert cli.main(["stages"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [line.split()[1] for line in lines] == [s.name for s in stages.STAGES]
    assert [s.number for s in stages.STAGES] == list(range(len(stages.STAGES)))


def test_a_stage_with_no_command_shows_by_hand(capsys):
    cli.main(["stages"])
    out = capsys.readouterr().out
    for s in stages.STAGES:
        line = next(line for line in out.splitlines() if line.split()[1] == s.name)
        assert (s.command or "by hand") in line


def test_version(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_is_an_error():
    with pytest.raises(SystemExit) as stop:
        cli.main([])
    assert stop.value.code == 2
