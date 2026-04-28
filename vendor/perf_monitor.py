from __future__ import annotations

import csv
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings


@dataclass
class PerfMonitorConfig:
    enabled: bool = False
    output_path: str | None = None
    format: str = "csv"
    print_summary: bool = True
    collect_gpu: bool = True
    sync_cuda: bool = False
    detail_level: str = "coarse"


class PerfMonitor:
    FIELDNAMES = [
        "phase",
        "date",
        "wall_ms",
        "cpu_ms",
        "gpu_ms",
        "count",
    ]
    DETAIL_ORDER = {"coarse": 0, "standard": 1, "full": 2}

    def __init__(self, config: PerfMonitorConfig):
        self.config = config
        self.enabled = bool(config.enabled)
        self.output_path = Path(config.output_path).expanduser().resolve() if config.output_path else None
        self._started_at = time.perf_counter()
        self._closed = False
        self._writer = None
        self._file = None
        self._torch = None
        self._torch_cuda = None
        self.detail_level = (config.detail_level or "coarse").strip().lower()
        if self.detail_level not in self.DETAIL_ORDER:
            raise ValueError(f"unsupported detail_level: {self.detail_level}")

        if not self.enabled:
            return
        if self.config.format != "csv":
            raise ValueError(f"unsupported perf monitor format: {self.config.format}")
        if self.output_path is None:
            raise ValueError("perf monitor output_path is required when monitor is enabled")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._init_gpu()
        print(f"[PERF] enabled output={self.output_path}")

    @classmethod
    def from_config(cls, organize_config: dict[str, Any]) -> "PerfMonitor":
        monitor_config = dict(organize_config.get("monitor", {}))
        if monitor_config.get("enabled") and not monitor_config.get("output_path"):
            monitor_config["output_path"] = str(Path(organize_config["constants"]["output_root"]) / "perf_metrics.csv")
        return cls(PerfMonitorConfig(**monitor_config))

    def section(self, event: str, date: int | None = None):
        if not self.enabled:
            return nullcontext()
        return self._section(event, date)

    def cuda_section(self, event: str, date: int | None = None):
        if not self.enabled:
            return nullcontext()
        if self._torch_cuda is None:
            return self._section(event, date)
        return self._cuda_section(event, date)

    def mixed_section(self, event: str, date: int | None = None):
        if not self.enabled:
            return nullcontext()
        if self._torch_cuda is None:
            return self._section(event, date)
        return self._mixed_section(event, date)

    def accumulator(self, event: str, date: int | None = None):
        if not self.enabled:
            return NullAccumulator()
        return PerfAccumulator(self, event, date)

    def detail_enabled(self, level: str) -> bool:
        return self.enabled and self.DETAIL_ORDER[self.detail_level] >= self.DETAIL_ORDER[level]

    def maybe_section(self, event: str, date: int | None = None, *, level: str = "full", kind: str = "cpu"):
        if not self.detail_enabled(level):
            return nullcontext()
        if kind == "cuda":
            return self.cuda_section(event, date=date)
        if kind == "mixed":
            return self.mixed_section(event, date=date)
        return self.section(event, date=date)

    @contextmanager
    def _section(self, event: str, date: int | None = None):
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        try:
            yield
        finally:
            self._sync_cuda_if_needed()
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            self._write_metric_row(
                phase=event,
                date=date,
                wall_ms=(end_wall - start_wall) * 1000.0,
                cpu_ms=(end_cpu - start_cpu) * 1000.0,
                gpu_ms=None,
                count=1,
            )

    @contextmanager
    def _cuda_section(self, event: str, date: int | None = None):
        if self._torch_cuda is None:
            with self._section(event, date):
                yield
            return
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        start_event = self._torch_cuda.Event(enable_timing=True)
        end_event = self._torch_cuda.Event(enable_timing=True)
        start_event.record()
        try:
            yield
        finally:
            end_event.record()
            try:
                self._torch_cuda.synchronize()
                gpu_ms = float(start_event.elapsed_time(end_event))
            except Exception as exc:
                warnings.warn(f"[PERF] cuda timing failed for {event}: {exc}", RuntimeWarning)
                gpu_ms = None
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            self._write_metric_row(
                phase=event,
                date=date,
                wall_ms=(end_wall - start_wall) * 1000.0,
                cpu_ms=None,
                gpu_ms=gpu_ms,
                count=1,
            )

    @contextmanager
    def _mixed_section(self, event: str, date: int | None = None):
        start_cpu = time.process_time()
        self._torch_cuda.synchronize()
        start_wall = time.perf_counter()
        try:
            yield
        finally:
            self._torch_cuda.synchronize()
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            self._write_metric_row(
                phase=event,
                date=date,
                wall_ms=(end_wall - start_wall) * 1000.0,
                cpu_ms=(end_cpu - start_cpu) * 1000.0,
                gpu_ms=None,
                count=1,
            )

    def close(self):
        if not self.enabled or self._closed:
            return
        self._closed = True
        total_wall_s = time.perf_counter() - self._started_at
        if self._file is not None:
            self._file.flush()
            self._file.close()
        if self.config.print_summary:
            print(f"[PERF] summary output={self.output_path} total_wall_s={total_wall_s:.3f}")

    def _write_row(self, row: dict[str, Any]):
        if self._writer is None or self._file is None:
            return
        self._writer.writerow(row)
        self._file.flush()

    def _init_gpu(self):
        if not self.config.collect_gpu:
            return
        try:
            import torch
        except Exception:
            return
        if not torch.cuda.is_available():
            return
        self._torch = torch
        self._torch_cuda = torch.cuda

    def _sync_cuda_if_needed(self):
        if self._torch_cuda is not None and self.config.sync_cuda:
            self._torch_cuda.synchronize()

    def _write_metric_row(
        self,
        *,
        phase: str,
        date: int | None,
        wall_ms: float | None,
        cpu_ms: float | None,
        gpu_ms: float | None,
        count: int,
    ):
        self._write_row(
            {
                "phase": phase,
                "date": "" if date is None else int(date),
                "wall_ms": self._format_ms(wall_ms),
                "cpu_ms": self._format_ms(cpu_ms),
                "gpu_ms": self._format_ms(gpu_ms),
                "count": int(count),
            }
        )

    @staticmethod
    def _format_ms(value: float | None) -> str:
        if value is None:
            return ""
        return f"{value:.3f}"


class PerfAccumulator:
    def __init__(self, monitor: PerfMonitor, event: str, date: int | None):
        self.monitor = monitor
        self.event = event
        self.date = date
        self.wall_ms = 0.0
        self.cpu_ms = 0.0
        self.gpu_ms = 0.0
        self.count = 0

    @contextmanager
    def tick(self):
        if not self.monitor.enabled:
            yield
            return
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        try:
            yield
        finally:
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            self.wall_ms += (end_wall - start_wall) * 1000.0
            self.cpu_ms += (end_cpu - start_cpu) * 1000.0
            self.count += 1

    def add(self, *, wall_ms: float = 0.0, cpu_ms: float = 0.0, gpu_ms: float = 0.0, count: int = 1):
        if not self.monitor.enabled:
            return
        self.wall_ms += wall_ms
        self.cpu_ms += cpu_ms
        self.gpu_ms += gpu_ms
        self.count += count

    def flush(self):
        if not self.monitor.enabled or self.count <= 0:
            return
        self.monitor._write_metric_row(
            phase=self.event,
            date=self.date,
            wall_ms=self.wall_ms,
            cpu_ms=self.cpu_ms if self.cpu_ms > 0 else None,
            gpu_ms=self.gpu_ms if self.gpu_ms > 0 else None,
            count=self.count,
        )


class NullAccumulator:
    @contextmanager
    def tick(self):
        yield

    def add(self, **kwargs):
        return None

    def flush(self):
        return None
