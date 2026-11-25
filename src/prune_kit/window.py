"""The window runner (plans/PLAN_PRUNE_KIT.md §5.7): run test servers in one window, and restore the machine.

A machine file (YAML) holds the facts of one machine: each worker with its stop, start, health and metrics, the
hours in which a worker must be up, and the memory limits of the watch. A window file (YAML) holds the points of one
window: each point has the command of its test server, a health URL, and the clients that run against it.

The rules:

- A window stops a live worker only when the command line names it (``--stop``), and only when its ``/metrics`` shows
  0 running and 0 waiting requests.
- The deadlines have no defaults. No point boots after ``boot_by``. Each client stops at ``restore_at``.
- A boot that is not healthy in 15 minutes goes down. After two failed boots in a row, the window ends.
- A watch reads the available memory and the free swap during each client. Below a limit, it stops the client.
- The window ends with the restore of each stopped worker: on success, on each failure, and at ``restore_at``.
"""

import json
import os
import re
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import yaml

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
BOOT_S, IDLE_S, RESTORE_S, POLL_S = 900, 1800, 1200, 15
SHELL = "/bin/bash"   # the commands of a machine file are bash: `kill -TERM -- -PID` is not valid in sh
BUSY = re.compile(r"^vllm:num_requests_(?:running|waiting)\{[^}]*\} (\S+)", re.M)


class WindowError(Exception):
    """A fault of a file, or a safety failure of the window. The safe action is the restore."""


def load(path, required: dict) -> dict:
    """A machine file or a window file. ``required`` gives the keys that each entry of the top key must have."""
    try:
        data = yaml.safe_load(Path(path).expanduser().read_text())
    except (OSError, yaml.YAMLError) as e:
        raise WindowError(f"cannot read {path}: {e}") from e
    for top, keys in required.items():
        entries = data.get(top) if isinstance(data, dict) else None
        if not entries:
            raise WindowError(f"{path}: the key {top} is missing or empty")
        for entry in entries.values() if isinstance(entries, dict) else entries:
            if missing := [k for k in keys if not (isinstance(entry, dict) and entry.get(k))]:
                raise WindowError(f"{path}: an entry of {top} has no {', '.join(missing)}")
    return data


def load_machine(path) -> dict:
    return load(path, {"workers": ("stop", "start", "health", "metrics")})


def load_window(path) -> list:
    """The points of a window file. The key ``release`` of the file is a list of folders: see ``System.release``."""
    return load(path, {"points": ("name", "boot", "health")})["points"]


def must_be_up(worker: dict, start: datetime, end: datetime) -> str | None:
    """The first span of the ``up`` hours of a worker that the time from ``start`` to ``end`` touches."""
    for day in range((end.date() - start.date()).days + 1):
        date = start.date() + timedelta(days=day)
        for span in worker.get("up", []):
            first, last = (datetime.combine(date, datetime.strptime(span[k], "%H:%M").time()) for k in ("from", "to"))
            if DAYS[date.weekday()] in span["days"] and start < last and end > first:
                return f"{DAYS[date.weekday()]} {span['from']} to {span['to']}"
    return None


class System:
    """The machine: commands, URLs, the clock and the memory. A test replaces it."""

    def now(self) -> datetime:
        return datetime.now()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def run(self, command: str) -> int:
        return subprocess.run(command, shell=True, executable=SHELL).returncode

    def spawn(self, command: str):
        return subprocess.Popen(command, shell=True, executable=SHELL, start_new_session=True)

    def done(self, process) -> int | None:
        return process.poll()

    def kill(self, process, patience: float = 60) -> None:
        """Stop the whole process group of a spawned command: SIGTERM, then SIGKILL after ``patience`` seconds. The
        shell of the command ends at once, but a server can run on for some seconds, and a server that loads its
        weights can ignore SIGTERM. So the group must be gone, and not only the shell."""
        for sig, seconds in ((signal.SIGTERM, patience), (signal.SIGKILL, 10)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            end = time.monotonic() + seconds
            while time.monotonic() < end and self.alive(process.pid):
                time.sleep(0.2)
        process.wait()

    @staticmethod
    def alive(group: int) -> bool:
        """Does the process group have a process that is not a zombie?"""
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                fields = stat.read_text().rsplit(")", 1)[1].split()
            except OSError:
                continue
            if int(fields[2]) == group and fields[0] != "Z":
                return True
        return False

    def get(self, url: str) -> str | None:
        try:
            with urllib.request.urlopen(url, timeout=5) as reply:
                return reply.read().decode(errors="replace")
        except OSError:
            return None

    def release(self, folders) -> None:
        """Ask the kernel to drop the cached pages of each file of ``folders``. A server that loads its weights
        with CUDA needs free memory: on a machine with unified memory, the page cache of a folder that a window
        wrote or read can make that start fail (the Qwen restore of 2026-11-25)."""
        os.sync()
        for folder in folders:
            for path in Path(folder).expanduser().rglob("*"):
                try:
                    if path.is_file() and path.stat().st_size >= 1 << 20:
                        fd = os.open(path, os.O_RDONLY)
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        os.close(fd)
                except OSError:
                    pass

    def memory(self) -> tuple[float, float]:
        """The available memory and the free swap, in GiB."""
        with open("/proc/meminfo") as f:
            kb = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return kb["MemAvailable"] / 2**20, kb["SwapFree"] / 2**20


class DryRun(System):
    """No command runs and no time passes: each command is a log line, each boot is healthy, and each client ends
    at once. A stop command of the machine file puts its worker down until its start command."""

    def __init__(self, log, machine: dict, start: datetime):
        self.log, self.clock, self.off, self.serving = log, start, set(), False
        self.live = {w[k] for w in machine["workers"].values() for k in ("health", "metrics")}
        self.effects = {w[k]: (k, w["health"]) for w in machine["workers"].values() for k in ("stop", "start")}

    now = lambda self: self.clock
    done = lambda self, process: 0
    memory = lambda self: (999.0, 999.0)

    def get(self, url):
        """A worker answers until its stop command. Each other URL is a test server: it answers after a spawn."""
        return None if url in self.off or (url not in self.live and not self.serving) else ""

    def kill(self, process):
        self.serving = False

    def release(self, folders):
        self.log(f"dry run: drop the cached pages of {', '.join(map(str, folders))}")

    def sleep(self, seconds):
        self.clock += timedelta(seconds=seconds)

    def run(self, command):
        self.log(f"dry run: {command}")
        kind, health = self.effects.get(command, (None, None))
        self.off = self.off | {health} if kind == "stop" else self.off - {health}
        return 0

    def spawn(self, command):
        self.log(f"dry run: {command}")
        self.serving = True
        return command


class Window:
    def __init__(self, machine: dict, stop, start_after: datetime, boot_by: datetime, restore_at: datetime,
                 system: System | None = None, log=print, results=None, release=()):
        self.machine, self.stop, self.sys, self.log = machine, list(stop), system or System(), log
        self.results, self.rows, self.lowest, self.release = results, [], float("inf"), list(release)
        self.start_after, self.boot_by, self.restore_at = start_after, boot_by, restore_at
        self.stopped, self.limits = [], machine.get("memory", {})

    def check(self) -> None:
        """Refuse a window with a wrong order of the deadlines, an unknown worker, or a stop in the up hours."""
        if not self.start_after < self.boot_by < self.restore_at:
            raise WindowError("the deadlines must be in the order start_after, boot_by, restore_at")
        if self.restore_at <= self.sys.now():
            raise WindowError("restore_at is in the past")
        for name in self.stop:
            if name not in self.machine["workers"]:
                raise WindowError(f"the machine file has no worker {name}")
            end = self.restore_at + timedelta(seconds=RESTORE_S)   # the worker is up again after the restore
            if span := must_be_up(self.machine["workers"][name], self.start_after, end):
                raise WindowError(f"the worker {name} must be up {span}: the window and its restore touch that time")

    def wait(self, ok, seconds: float | None = None) -> bool:
        """Poll ``ok`` until it is true, for at most ``seconds`` and not past ``restore_at``."""
        end = self.restore_at if seconds is None else min(self.sys.now() + timedelta(seconds=seconds), self.restore_at)
        while not (good := ok()) and self.sys.now() < end:
            self.sys.sleep(POLL_S)
        return bool(good)

    def idle(self, worker: dict) -> bool:
        text = self.sys.get(worker["metrics"])
        return text is not None and sum(float(v) for v in BUSY.findall(text)) == 0

    def stop_workers(self) -> None:
        for name in self.stop:
            worker = self.machine["workers"][name]
            if self.sys.get(worker["health"]) is None:
                self.log(f"the worker {name} is down already: the window does not restore it")
                continue
            if not self.wait(lambda: self.idle(worker), IDLE_S):
                raise WindowError(f"the worker {name} stayed busy: not stopped")
            self.log(f"stop the worker {name}")
            self.stopped.append(name)   # before the command: a stop that fails halfway still gets its restore
            if self.sys.run(worker["stop"]) != 0 or self.sys.get(worker["health"]) is not None:
                raise WindowError(f"the worker {name} did not stop")

    def boot(self, point: dict):
        """Start the test server of a point. Return its process when it is healthy, or None after a failed boot."""
        need = point.get("need_gib", 0)
        if self.release:
            self.sys.release(self.release)
        if not self.wait(lambda: self.sys.memory()[0] >= need, 300):
            raise WindowError(f"{point['name']}: less than {need} GiB available ({self.sys.memory()[0]:.1f})")
        if self.sys.get(point["health"]) is not None:
            raise WindowError(f"{point['name']}: a server answers on {point['health']} before the boot")
        self.log(f"boot {point['name']} ({self.sys.memory()[0]:.1f} GiB available)")
        process = self.sys.spawn(point["boot"])
        healthy = lambda: self.sys.get(point["health"]) is not None or self.sys.done(process) is not None
        if self.wait(healthy, BOOT_S) and self.sys.get(point["health"]) is not None:
            return process
        self.log(f"{point['name']}: not healthy: down")
        self.down(point, process)
        return None

    def down(self, point: dict, process) -> None:
        """Stop the test server of a point. The health URL must stop its answers: a server that still answers would
        look like the next point."""
        self.sys.kill(process)
        if point.get("down"):
            self.sys.run(point["down"])
        end = self.sys.now() + timedelta(seconds=120)
        while self.sys.get(point["health"]) is not None:
            if self.sys.now() >= end:
                raise WindowError(f"{point['name']}: the test server still answers after its stop")
            self.sys.sleep(POLL_S)

    def client(self, command: str) -> int | None:
        """Run a client with the memory watch. Return its exit code, or None when the watch or the deadline
        stopped it."""
        self.log(f"client: {command}")
        process, low = self.sys.spawn(command), None

        def over():
            nonlocal low
            available, swap = self.sys.memory()
            self.lowest = min(self.lowest, available)
            if available < self.limits.get("min_available_gib", 1) or swap < self.limits.get("min_free_swap_gib", 0):
                low = f"{available:.1f} GiB available, {swap:.1f} GiB of swap free"
            return low or self.sys.done(process) is not None

        self.wait(over)
        code = self.sys.done(process)
        if low or code is None:
            self.log(f"stop the client: {low or 'restore_at is here'}")
            self.sys.kill(process)
            return None
        return code

    def restore(self) -> bool:
        """Start each stopped worker again, the last one first. Return True when each one is healthy."""
        good = True
        for name in reversed(self.stopped):
            worker = self.machine["workers"][name]
            for attempt in (1, 2):
                self.log(f"restore the worker {name} (try {attempt})")
                if self.release:
                    self.sys.release(self.release)
                started = self.sys.run(worker["start"]) == 0
                end = self.sys.now() + timedelta(seconds=RESTORE_S if started else 60)   # failed: try again soon
                while (up := self.sys.get(worker["health"]) is not None) is False and self.sys.now() < end:
                    self.sys.sleep(POLL_S)
                if up:
                    break
            self.log(f"RESTORE {'DONE' if up else 'FAIL'}: the worker {name}")
            good = good and up
        return good

    def run(self, points: list) -> int:
        """Exit code 0: each point ran. 1: a point failed or did not run. 3: the restore failed."""
        code, fails = 0, 0
        try:
            self.check()
            self.sys.sleep(max(0.0, (self.start_after - self.sys.now()).total_seconds()))
            self.stop_workers()
            for point in points:
                if self.sys.now() >= self.boot_by or fails >= 2:
                    why = "two failed boots in a row" if fails >= 2 else "after boot_by"
                    self.log(f"{point['name']}: not booted: {why}")
                    code = 1
                    continue
                before, self.lowest = self.sys.memory()[0], float("inf")
                process, codes = self.boot(point), [None]
                fails = 0 if process else fails + 1
                if process:
                    try:
                        codes = [self.client(c) for c in point.get("clients", [])]
                    finally:
                        self.down(point, process)
                code = code if all(c == 0 for c in codes) else 1
                self.rows.append({"name": point["name"], "booted": bool(process), "clients": codes,
                                  "available_before_gib": round(before, 2),
                                  "available_lowest_gib": round(min(self.lowest, before), 2)})
        except Exception as e:   # each failure ends with the safe action
            self.log(f"failure: {e}: the safe action, the restore")
            code = 1
        restored = self.restore()
        if self.results:   # the points that booted or failed, for the report of a probe
            Path(self.results).write_text(json.dumps({"points": self.rows, "restored": restored}, indent=1))
        return code if restored else 3
