"""Bounded reuse of exact native decoded RGB frames, before model transforms."""
from collections import OrderedDict
import threading


class DecodedFrameCache:
    """An LRU byte bound on retained RGB payloads, not total process memory.

    Returned images are copies. A model transform cannot mutate cached bytes.
    Oversized frames pass through without being retained. The original native
    decoder still checks source presentation timestamps and color conversion.
    """
    def __init__(self, max_bytes: int, *, max_entries=4096):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._items = OrderedDict()
        self._lock = threading.RLock()
        self._bytes = 0
        self._peak_bytes = 0
        self._hits = self._misses = self._evictions = 0

    def decode(self, video, wanted, canonical=False, *, decoder):
        wanted = set(wanted)
        # Match the native decoder's input checks even on a full cache hit.
        if wanted and (min(wanted) < 0 or max(wanted) >= len(video.pts_us)):
            return decoder(video, wanted, canonical=canonical)
        result, missing = {}, set()
        with self._lock:
            for index in wanted:
                key = (video.sha256, index, bool(canonical), "native-rgb-v1")
                if key in self._items:
                    from PIL import Image
                    payload, dimensions = self._items[key]
                    self._items.move_to_end(key)
                    result[index] = Image.frombytes("RGB", dimensions, payload)
                    self._hits += 1
                else:
                    missing.add(index)
                    self._misses += 1
        if missing:
            decoded = decoder(video, missing, canonical=canonical)
            if set(decoded) != missing:
                raise ValueError("The native decoder did not return every requested frame")
            with self._lock:
                for index, image in decoded.items():
                    if image.mode != "RGB":
                        raise ValueError("The decoded-frame cache requires native RGB images")
                    size = image.width * image.height * 3
                    key = (video.sha256, index, bool(canonical), "native-rgb-v1")
                    if 0 < size <= self.max_bytes:
                        if key in self._items:
                            self._bytes -= len(self._items.pop(key)[0])
                        while self._items and (self._bytes + size > self.max_bytes or len(self._items) >= self.max_entries):
                            _, (removed, _) = self._items.popitem(last=False)
                            self._bytes -= len(removed)
                            self._evictions += 1
                        payload = image.tobytes()
                        if len(payload) != size:
                            raise ValueError("Native RGB payload size changed")
                        self._items[key] = (payload, (image.width, image.height))
                        self._bytes += size
                        self._peak_bytes = max(self._peak_bytes, self._bytes)
                    result[index] = image
        return result

    def clear(self):
        with self._lock:
            self._items.clear()
            self._bytes = 0

    def stats(self):
        """Return cumulative cache-lifetime counters, including after clear()."""
        with self._lock:
            return dict(retained_bytes=self._bytes, peak_retained_bytes=self._peak_bytes, max_bytes=self.max_bytes,
                        entries=len(self._items), max_entries=self.max_entries, hits=self._hits,
                        misses=self._misses, evictions=self._evictions)
