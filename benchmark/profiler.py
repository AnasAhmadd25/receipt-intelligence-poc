"""Process-tree memory sampling and OS-accounted core seconds, not provisioned CPU."""
import os
import platform
import resource
import threading
import time
from contextlib import AbstractContextManager

import psutil


def cpu_seconds():
    own = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime


class Profile(AbstractContextManager):
    def __init__(self, interval=0.05):
        self.interval = interval
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.samples = []
        self.last_cpu = {}
        self.sampled_cpu = []
        self.result = {}

    def sample(self):
        members = [self.process]
        try:
            members += self.process.children(recursive=True)
        except psutil.Error:
            pass
        rss = 0
        percent = 0.0
        now = time.perf_counter()
        for process in members:
            try:
                rss += process.memory_info().rss
                cpu = process.cpu_times()
                total = cpu.user + cpu.system
                previous = self.last_cpu.get(process.pid)
                if previous and now > previous[0]:
                    percent += max(0, total - previous[1]) / (now - previous[0]) * 100
                self.last_cpu[process.pid] = (now, total)
            except psutil.Error:
                pass
        self.sampled_cpu.append(percent)
        self.samples.append((time.perf_counter(), rss, cpu_seconds()))

    def _loop(self):
        while not self.stop.wait(self.interval):
            self.sample()

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_cpu = cpu_seconds()
        self.sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()
        self.sample()
        wall = time.perf_counter() - self.start_time
        core_seconds = max(0.0, cpu_seconds() - self.start_cpu)
        baseline = self.samples[0][1]
        peak = max(x[1] for x in self.samples)
        peaks = [max(0, (b[2] - a[2]) / (b[0] - a[0]) * 100) for a, b in zip(self.samples, self.samples[1:]) if b[0] > a[0]]
        self.result = {
            "wall_seconds": wall,
            "cpu_core_seconds": core_seconds,
            "average_cpu_percent": core_seconds / wall * 100 if wall else 0,
            "peak_cpu_percent": max(self.sampled_cpu, default=0),
            "baseline_ram_mib": baseline / 2**20,
            "peak_ram_mib": peak / 2**20,
            "incremental_ram_mib": max(0, peak - baseline) / 2**20,
            "sample_interval_seconds": self.interval,
            "scope": "local process tree, includes provider waiting; CPU 100%=one core",
            "peak_cpu_caveat": "50ms live process-tree samples may miss short-lived children; average/core-seconds use OS waited-child accounting",
            "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "system_ram_bytes": psutil.virtual_memory().total,
            "platform": platform.platform(),
        }
        return False
