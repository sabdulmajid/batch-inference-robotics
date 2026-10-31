from types import SimpleNamespace
import importlib.util
import unittest

from bir.frame_cache import DecodedFrameCache


@unittest.skipUnless(importlib.util.find_spec("PIL"), "requires optional Pillow CPU dependency")
class FrameCacheTests(unittest.TestCase):
    def test_exact_keys_copy_isolation_and_bound(self):
        from PIL import Image
        video = SimpleNamespace(sha256="one", pts_us=(0, 10, 20))
        calls = []
        def decode(video, wanted, canonical=False):
            calls.append((video.sha256, wanted, canonical))
            return {i: Image.new("RGB", (2, 2), (1 if video.sha256 == "one" else 2, i, int(canonical))) for i in wanted}
        cache = DecodedFrameCache(24)
        first = cache.decode(video, {0, 1}, decoder=decode)
        first[0].putpixel((0, 0), (9, 9, 9))
        hit = cache.decode(video, {0}, decoder=decode)
        self.assertEqual(hit[0].getpixel((0, 0)), (1, 0, 0))
        cache.decode(video, {0}, canonical=True, decoder=decode)
        cache.decode(video, {2}, decoder=decode)
        self.assertEqual(cache.stats()["retained_bytes"], 24)
        self.assertGreater(cache.stats()["evictions"], 0)
        self.assertEqual(len(calls), 3)
        video.sha256 = "two"
        cache.decode(video, {0}, decoder=decode)
        self.assertEqual(len(calls), 4)

    def test_oversized_frames_are_not_retained(self):
        from PIL import Image
        video = SimpleNamespace(sha256="one", pts_us=(0,))
        cache = DecodedFrameCache(1)
        result = cache.decode(video, {0}, decoder=lambda v, w, canonical=False: {0: Image.new("RGB", (2, 2), (1, 0, 0))})
        self.assertEqual(result[0].getpixel((0, 0)), (1, 0, 0))
        self.assertEqual(cache.stats()["retained_bytes"], 0)

    def test_entry_limit_bounds_metadata_count(self):
        from PIL import Image
        video = SimpleNamespace(sha256="one", pts_us=(0, 1, 2))
        cache = DecodedFrameCache(1000, max_entries=1)
        cache.decode(video, {0, 1, 2}, decoder=lambda v, w, canonical=False:
                     {i: Image.new("RGB", (1, 1)) for i in w})
        self.assertEqual(cache.stats()["entries"], 1)
        self.assertEqual(cache.stats()["retained_bytes"], 3)


if __name__ == "__main__":
    unittest.main()
