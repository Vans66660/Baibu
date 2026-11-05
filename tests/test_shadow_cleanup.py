import sys
import unittest
from pathlib import Path
from unittest.mock import Mock
import cv2
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class ShadowCleanupTests(unittest.TestCase):
    def scene(self):
        yy, xx = np.mgrid[:240, :320]
        ground = np.stack([205 + xx*.04 + yy*.03, 195 + xx*.04 + yy*.03, 180 + xx*.04 + yy*.03], axis=-1).astype(np.uint8)
        arr = ground.copy()
        shadow = np.zeros((240, 320), np.uint8)
        cv2.ellipse(shadow, (184, 179), (49, 13), 0, 0, 360, 255, -1)
        arr[shadow > 0] = (arr[shadow > 0].astype(float)*.76).astype(np.uint8)
        mask = np.zeros((240, 320), np.uint8); mask[70:180, 110:170] = 255
        arr[mask > 0] = [40, 90, 160]
        return ground, arr, mask, shadow

    def test_shadow_is_removed_beyond_object_without_growing_product(self):
        ground, arr, mask, shadow = self.scene()
        removal = app.object_repair_mask(arr, mask)
        self.assertGreater(np.mean(removal[shadow > 0] > 0), .95)
        lama = Mock(side_effect=AssertionError('smooth background should not need model'))
        clean, product, crop = app.compose_layers(Image.fromarray(arr), mask, lama)
        self.assertEqual(crop, [110, 70, 60, 110])
        self.assertEqual(product.size, (60, 110))
        self.assertLess(np.mean(np.abs(np.asarray(clean)[shadow > 0].astype(float)-ground[shadow > 0])), 2)
        self.assertTrue(np.array_equal(np.asarray(clean)[10:30, 10:30], arr[10:30, 10:30]))

    def test_disconnected_dark_object_and_colored_neighbor_are_preserved(self):
        _, arr, mask, _ = self.scene()
        arr[60:80, 185:200] = [35, 35, 35]
        arr[100:130, 170:185] = [210, 20, 30]
        removal = app.object_repair_mask(arr, mask)
        self.assertFalse(np.any(removal[62:78, 187:198]))
        self.assertFalse(np.any(removal[104:125, 181:184]))

    def test_textured_background_falls_back_to_model(self):
        rng = np.random.default_rng(42)
        arr = rng.integers(40, 230, (180, 200, 3), dtype=np.uint8)
        mask = np.zeros((180,200), np.uint8);mask[70:110, 80:120] = 255
        lama = Mock(side_effect=lambda image, _: image)
        app.compose_layers(Image.fromarray(arr),mask,lama)
        lama.assert_called_once()

    def test_mask_ranking_avoids_partial_and_leaking_masks(self):
        candidates = np.zeros((3,100,100),bool)
        candidates[0,40:60,40:60]=True
        candidates[1,20:80,20:80]=True
        candidates[2,5:95,5:95]=True
        self.assertEqual(app.choose_mask_index(candidates,[.91,.87,.94],[20,20,80,80]),1)

if __name__ == '__main__': unittest.main()
