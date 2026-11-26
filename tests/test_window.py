from datetime import datetime, timedelta
from pathlib import Path

import pytest

from prune_moe_kit import cli, window

T0 = datetime(2026, 11, 28, 22, 0)   # a Saturday
at = lambda minutes: T0 + timedelta(minutes=minutes)
MACHINE = {"workers": {
    "deepseek": {"stop": "stop-ds", "start": "start-ds", "health": "ds/health", "metrics": "ds/metrics",
                 "up": [{"days": ["mon", "tue", "wed", "thu", "fri"], "from": "05:00", "to": "16:00"}]},
    "qwen": {"stop": "stop-qwen", "start": "start-qwen", "health": "qwen/health", "metrics": "qwen/metrics"}},
    "memory": {"min_available_gib": 2, "min_free_swap_gib": 1}}
POINTS = [{"name": "full", "boot": "serve-full", "health": "test/health", "need_gib": 90, "down": "down-full",
           "clients": ["capture", "score"]},
          {"name": "k208", "boot": "serve-k208", "health": "test/health", "clients": ["score"]}]


class Fake(window.System):
    """A machine with its own clock. ``boots``: does a test server get healthy? ``client_s``: the time of a client."""

    def __init__(self, busy=0.0, boots=True, client_s=60, restores=True):
        self.clock, self.log, self.up, self.busy = T0, [], {"ds/health", "qwen/health"}, busy
        self.boots, self.client_s, self.restores, self.mem = boots, client_s, restores, (100.0, 8.0)

    now = lambda self: self.clock
    memory = lambda self: self.mem(self.clock) if callable(self.mem) else self.mem

    def sleep(self, seconds):
        self.clock += timedelta(seconds=seconds)

    def run(self, command):
        self.log.append(command)
        kind, _, name = command.partition("-")
        url = {"ds": "ds/health", "qwen": "qwen/health"}.get(name)
        if kind == "stop":
            self.up.discard(url)
        elif kind == "start" and self.restores:
            self.up.add(url)
        return 0

    def spawn(self, command):
        self.log.append(command)
        server = command.startswith("serve")
        if server and self.boots:
            self.up.add("test/health")
        return {"command": command, "ends": None if server else self.clock + timedelta(seconds=self.client_s)}

    done = lambda self, process: 0 if process["ends"] and self.clock >= process["ends"] else None

    def kill(self, process):
        self.log.append(f"kill {process['command']}")
        self.up.discard("test/health")

    def get(self, url):
        if url.endswith("/metrics"):
            return (f'# HELP x\nvllm:num_requests_running{{model_name="m"}} {self.busy}\n'
                    'vllm:num_requests_waiting{model_name="m"} 0.0\n'
                    'vllm:num_requests_waiting_by_reason{reason="x"} 9.0\n')
        return "" if url in self.up else None


def run(fake, stop=("deepseek",), points=POINTS, boot_by=120, restore_at=240):
    lines = []
    w = window.Window(MACHINE, stop, at(-1), at(boot_by), at(restore_at), fake, lines.append)
    return w.run(points), lines


def test_a_window_stops_the_named_worker_runs_each_point_and_restores():
    fake = Fake()
    code, lines = run(fake)
    assert code == 0 and fake.log == ["stop-ds", "serve-full", "capture", "score", "kill serve-full", "down-full",
                                      "serve-k208", "score", "kill serve-k208", "start-ds"]
    assert "RESTORE DONE: the worker deepseek" in lines[-1] and fake.up == {"ds/health", "qwen/health"}


def test_a_busy_worker_is_not_stopped():
    fake = Fake(busy=1.0)
    code, lines = run(fake)
    assert code == 1 and fake.log == [] and fake.clock >= at(30)
    assert "the worker deepseek stayed busy: not stopped" in lines[0]


def test_two_failed_boots_in_a_row_end_the_points_and_the_restore_runs():
    fake = Fake(boots=False)
    code, lines = run(fake, points=POINTS + [{"name": "k192", "boot": "serve-k192", "health": "test/health"}])
    assert code == 1 and fake.log == ["stop-ds", "serve-full", "kill serve-full", "down-full", "serve-k208",
                                      "kill serve-k208", "start-ds"]
    assert "k192: not booted: two failed boots in a row" in lines and at(30) <= fake.clock < at(33)


def test_no_boot_after_boot_by_and_a_client_stops_at_restore_at():
    fake = Fake(client_s=100 * 60)   # the first client ends after 100 minutes, and the second one has 20 minutes
    code, lines = run(fake, boot_by=90, restore_at=120)
    assert code == 1 and "stop the client: restore_at is here" in lines and "k208: not booted: after boot_by" in lines
    assert fake.log == ["stop-ds", "serve-full", "capture", "score", "kill score", "kill serve-full", "down-full",
                        "start-ds"]


def test_the_watch_stops_a_client_below_a_memory_limit():
    fake = Fake(client_s=600)
    fake.mem = lambda clock: (100.0, 8.0) if clock < at(2) else (1.5, 8.0)
    code, lines = run(fake, points=POINTS[1:])
    assert code == 1 and "stop the client: 1.5 GiB available, 8.0 GiB of swap free" in lines
    assert fake.log[-3:] == ["kill score", "kill serve-k208", "start-ds"]


def test_low_memory_before_a_boot_and_a_failed_restore():
    fake = Fake(restores=False)
    fake.mem = (40.0, 8.0)
    code, lines = run(fake)
    assert code == 3 and "failure: full: less than 90 GiB available (40.0): the safe action, the restore" in lines
    assert fake.log == ["stop-ds", "start-ds", "start-ds"] and lines[-1] == "RESTORE FAIL: the worker deepseek"


def test_a_worker_that_is_down_already_is_not_restored_and_no_worker_stops_without_its_name():
    fake = Fake()
    fake.up.discard("qwen/health")
    assert run(fake, stop=("qwen",), points=POINTS[1:])[0] == 0 and "start-qwen" not in fake.log
    fake = Fake()
    assert run(fake, stop=())[0] == 0 and not [c for c in fake.log if c.startswith(("stop", "start"))]


@pytest.mark.parametrize("kwargs, fault", [
    ({"boot_by": 300}, "the deadlines must be in the order"), ({"stop": ("llama",)}, "has no worker llama"),
    ({"restore_at": 31 * 60 - 10}, "the worker deepseek must be up mon 05:00 to 16:00")])   # Monday 04:50 + restore
def test_check_refuses_a_wrong_window_before_a_command_runs(kwargs, fault):
    fake = Fake()
    code, lines = run(fake, **kwargs)
    assert code == 1 and fault in lines[0] and fake.log == []


def test_the_up_hours_and_the_files(tmp_path):
    ds = MACHINE["workers"]["deepseek"]
    assert window.must_be_up(ds, at(0), at(24 * 60)) is None                      # Saturday to Sunday night
    assert window.must_be_up(ds, at(0), at(31 * 60 + 1)) == "mon 05:00 to 16:00"  # to Monday 05:01
    assert window.must_be_up(MACHINE["workers"]["qwen"], at(0), at(9999)) is None
    path = tmp_path / "machine.yml"
    path.write_text("workers:\n  qwen: {stop: a, start: b, health: c}\n")
    with pytest.raises(window.WindowError, match="an entry of workers has no metrics"):
        window.load_machine(path)
    path.write_text("points:\n  - {name: full, boot: serve, health: url}\n")
    assert window.load_window(path)[0]["boot"] == "serve"
    with pytest.raises(window.WindowError, match="the key workers is missing"):
        window.load_machine(path)
    with pytest.raises(window.WindowError, match="cannot read"):
        window.load_machine(tmp_path / "none.yml")


def test_the_dry_run_of_the_command_prints_each_command_and_runs_none(capsys):
    examples = Path(__file__).parent.parent / "examples"
    args = ["window", str(examples / "window-qwen-score.yml"), "--machine", str(examples / "machine-spark.yml"),
            "--start-after", "2099-01-03 22:00", "--boot-by", "2099-01-04 01:00", "--dry-run"]
    assert cli.main([*args, "--restore-at", "2099-01-04 03:00", "--stop", "qwen"]) == 0
    out = capsys.readouterr().out
    order = ["stop the worker qwen", "dry run: p=$(cat", "boot full", "dry run: vllm serve unsloth/",
             "dry run: prune-kit gate --record", "boot k208", "dry run: vllm serve runs/qwen-k208",
             "restore the worker qwen", "dry run: python3 -I", "RESTORE DONE: the worker qwen"]
    places = [out.index(text) for text in order]
    assert places == sorted(places) and out.count("dry run: prune-kit gate --score") == 2 and "deepseek" not in out
    assert cli.main([*args, "--restore-at", "2099-01-04 01:10", "--stop", "deepseek"]) == 1   # the night-train job
    assert "the worker deepseek must be up sun 00:30 to 01:00" in capsys.readouterr().out
    assert cli.main([*args, "--restore-at", "soon"]) == 2
    assert cli.main(["window", "none.yml", *args[2:], "--restore-at", "2099-01-04 03:00"]) == 2


def test_a_stop_of_the_runner_stops_the_client_and_the_server_and_restores(tmp_path):
    fake = Fake(client_s=600)
    sleep = fake.sleep

    def stopped(seconds):   # the stop comes in the first client, after 2 minutes
        sleep(seconds)
        if fake.clock >= at(2) and "start-ds" not in fake.log and not fake.log[-1].startswith("kill"):
            raise KeyboardInterrupt

    fake.sleep = stopped
    points = [dict(POINTS[0], results=str(tmp_path / "a.json")), dict(POINTS[1], results=str(tmp_path / "b.json"))]
    lines = []
    w = window.Window(MACHINE, ["deepseek"], at(-1), at(120), at(240), fake, lines.append, tmp_path / "all.json")
    assert w.run(points) == 1
    assert fake.log == ["stop-ds", "serve-full", "capture", "kill capture", "kill serve-full", "down-full", "start-ds"]
    assert "failure: KeyboardInterrupt: the safe action, the restore" in lines[-3] and fake.up >= {"ds/health"}
    assert not (tmp_path / "a.json").exists() and '"restored": true' in (tmp_path / "all.json").read_text()


def test_each_point_can_name_its_results_file(tmp_path):
    import json
    points = [dict(POINTS[0], results=str(tmp_path / "a.json")), dict(POINTS[1], results=str(tmp_path / "b.json"))]
    w = window.Window(MACHINE, ["deepseek"], at(-1), at(120), at(240), Fake(), lambda _: None, tmp_path / "all.json")
    assert w.run(points) == 0
    names = lambda file: [p["name"] for p in json.loads((tmp_path / file).read_text())["points"]]
    assert (names("a.json"), names("b.json"), names("all.json")) == (["full"], ["k208"], ["full", "k208"])


def test_a_signal_becomes_a_failure_of_the_window_and_a_second_signal_has_no_effect():
    import os
    import signal
    before = {sig: signal.getsignal(sig) for sig in window.STOPS}
    try:
        window.on_stop()
        with pytest.raises(window.WindowError, match="the runner got SIGTERM"):
            os.kill(os.getpid(), signal.SIGTERM)
            signal.pause()
        os.kill(os.getpid(), signal.SIGINT)   # ignored: the restore must end
        assert all(signal.getsignal(sig) is signal.SIG_IGN for sig in window.STOPS)
    finally:
        for sig, handler in before.items():
            signal.signal(sig, handler)


def test_the_real_system_runs_bash_and_stops_the_whole_process_group(tmp_path):
    system = window.System()
    assert system.run("[[ a == a ]] && kill -0 -- -$(ps -o pgid= -p $$ | tr -d ' ')") == 0   # not valid in sh
    process = system.spawn(f"sleep 60 & echo $! > {tmp_path}/child; wait")
    while not (tmp_path / "child").exists() or not (tmp_path / "child").read_text().strip():
        system.sleep(0.05)
    child = Path(f"/proc/{int((tmp_path / 'child').read_text())}")
    assert system.done(process) is None and child.exists()
    system.kill(process)
    for _ in range(40):
        if not child.exists():
            break
        system.sleep(0.05)
    assert system.done(process) is not None and not child.exists()       # the child of the shell is gone too
    available, swap = system.memory()
    assert available > 0 and swap >= 0 and system.get("http://127.0.0.1:9/none") is None


def test_a_server_that_answers_after_its_stop_delays_the_next_point():
    fake = Fake()
    fake.kill = lambda process: fake.log.append(f"kill {process['command']}") or setattr(fake, "dying", fake.clock)
    get = fake.get
    fake.get = lambda url: None if url == "test/health" and getattr(fake, "dying", None) and (
        fake.clock - fake.dying).total_seconds() >= 40 and not fake.log[-1].startswith("serve") else get(url)
    code, lines = run(fake)
    boots = [n for n, c in enumerate(fake.log) if c.startswith("serve")]
    plain = Fake()
    run(plain)
    assert code == 0 and len(boots) == 2 and fake.log[boots[1] - 1] == "down-full"
    assert fake.clock - plain.clock >= timedelta(seconds=60)   # each stop waited until the health URL was silent


def test_a_server_on_the_test_port_before_a_boot_is_a_safety_failure():
    fake = Fake()
    fake.up.add("test/health")
    code, lines = run(fake)
    assert code == 1 and "full: a server answers on test/health before the boot" in lines[1]
    assert fake.log == ["stop-ds", "start-ds"]


def test_the_real_system_stops_a_process_that_ignores_sigterm(tmp_path):
    system = window.System()
    process = system.spawn(f"(trap '' TERM; echo $BASHPID > {tmp_path}/child; sleep 60) & wait")
    while not (tmp_path / "child").exists() or not (tmp_path / "child").read_text().strip():
        system.sleep(0.05)
    child = int((tmp_path / "child").read_text())
    assert window.System.alive(process.pid)
    system.kill(process, patience=1)
    assert not window.System.alive(process.pid)
    for _ in range(40):   # the child has a new parent after the stop: its entry stays until that parent reads its end
        if not Path(f"/proc/{child}").exists():
            break
        system.sleep(0.05)
    assert not Path(f"/proc/{child}").exists()


def test_a_failed_start_gets_its_second_try_soon_and_the_cache_is_released():
    fake, released, codes = Fake(restores=False), [], iter([1, 0])   # the first start command fails
    fake.release = released.append
    run_command, get = fake.run, fake.get
    start = lambda command: fake.log.append(command) or next(codes)
    fake.run = lambda command: start(command) if command == "start-ds" else run_command(command)
    fake.get = lambda url: "" if url == "ds/health" and fake.log.count("start-ds") == 2 else get(url)
    lines = []
    w = window.Window(MACHINE, ["deepseek"], at(-1), at(120), at(240), fake, lines.append, release=["/data/run"])
    assert w.run(POINTS[1:]) == 0 and fake.log.count("start-ds") == 2
    assert released == [["/data/run"]] * 3                 # before the boot, and before each start
    assert "restore the worker deepseek (try 2)" in lines and fake.clock - T0 < timedelta(minutes=10)


def test_release_drops_the_cached_pages_of_large_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/big.bin").write_bytes(b"x" * (2 << 20))
    (tmp_path / "small.txt").write_text("x")
    window.System().release([tmp_path, tmp_path / "none"])          # no error for a folder that does not exist
    assert (tmp_path / "sub/big.bin").stat().st_size == 2 << 20
