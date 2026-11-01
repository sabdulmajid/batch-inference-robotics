"""Optional CPU/wall spans for native preparation; no CUDA or heavy imports.

Nested spans and work on different threads are not additive wall time. The
coordinator's cProfile run attributes the unchanged upstream generation helper.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import threading
import time


class NativeProfile:
    def __init__(self, enabled=False, *, sink=None):
        self.enabled = bool(enabled)
        self.sink = sink

    @contextmanager
    def span(self, category, **identity):
        if not self.enabled:
            yield
            return
        started = time.perf_counter()
        record = dict(category=category, **identity, thread_id=threading.get_ident(),
                      start_monotonic_s=started, status='complete')
        try:
            yield
        except BaseException as error:
            record.update(status='failed', error_type=type(error).__name__)
            raise
        finally:
            ended = time.perf_counter()
            record.update(end_monotonic_s=ended, elapsed_s=ended - started,
                          utc=datetime.now(timezone.utc).isoformat())
            if self.sink is not None:
                self.sink(record)
            else:
                print('BIR_PROFILE ' + json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
