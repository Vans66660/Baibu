import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class CacheTests(unittest.TestCase):
    def test_lru_capacity_expiry_and_oversize(self):
        now = [0]
        cache = app.MemoryCache(10, 2, ttl=5, clock=lambda: now[0])
        cache.put('a', 'A', 4)
        cache.put('b', 'B', 4)
        self.assertEqual(cache.get('a'), 'A')
        cache.put('c', 'C', 4)
        self.assertIsNone(cache.get('b'))
        cache.put('large', 'L', 11)
        self.assertIsNone(cache.get('large'))
        now[0] = 6
        self.assertIsNone(cache.get('a'))
        self.assertEqual(cache.bytes, 0)

    def test_keys_distinguish_selection_and_reject_invalid_coords(self):
        self.assertEqual(app.selection_key('1,2,30,40', '', ''),
                         app.selection_key('1.0,2.00,30,40', '', 'ignored'))
        self.assertNotEqual(app.selection_key('1,2,30,40', '', ''),
                            app.selection_key('1,2,31,40', '', ''))
        for text in ['NaN,0,10,10', '0,0,inf,10', '1,2,3']:
            with self.assertRaises(ValueError):
                app.selection_key(text, '', '')

    def test_result_reuse_skips_models_and_separates_images(self):
        image = Image.new('RGB', (64, 64), 'white')
        with patch.object(app, 'result_cache', app.MemoryCache(1_000_000, 8)), \
             patch.object(app, 'load_models', return_value={'lama': None, 'device': 'cpu'}) as load, \
             patch.object(app, 'segment', return_value=np.ones((64, 64), dtype=np.uint8)), \
             patch.object(app, 'compose_layers', return_value=(image, image, [0, 0, 30, 30])):
            first = app.prepare_layers(image, '0,0,30,30', '', '')
            with app.model_lock:  # Cached results remain available during inference.
                repeat = app.prepare_layers(image, '0.0,0,30,30', '', '')
            self.assertFalse(first['cache']['result'])
            self.assertTrue(repeat['cache']['result'])
            self.assertEqual(first['clean'], repeat['clean'])
            self.assertEqual(load.call_count, 1)
            self.assertNotIn('background_ms', repeat['timings_ms'])
            self.assertGreaterEqual(repeat['timings_ms']['total_ms'], 0)
            app.prepare_layers(image, '0,0,31,30', '', '')
            app.prepare_layers(Image.new('RGB', (64, 64), 'red'), '0,0,30,30', '', '')
            self.assertEqual(load.call_count, 3)

    def test_failed_result_is_not_cached_and_lock_released(self):
        image = Image.new('RGB', (64, 64))
        with patch.object(app, 'result_cache', app.MemoryCache(1_000_000, 8)), \
             patch.object(app, 'load_models', side_effect=ValueError('test failure')) as load:
            for _ in range(2):
                with self.assertRaises(app.HTTPException):
                    app.prepare_layers(image, '0,0,30,30', '', '')
                self.assertFalse(app.model_lock.locked())
            self.assertEqual(load.call_count, 2)


if __name__ == '__main__':
    unittest.main()
