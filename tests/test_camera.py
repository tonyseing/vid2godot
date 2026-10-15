import unittest

import numpy as np

from anim.camera import SideCamera


class CameraTests(unittest.TestCase):
    def test_project_and_unproject_round_trip(self):
        cam = SideCamera(1280, 720, 400.0, 0.1, 0.3)
        xz = np.array([[0.1, 0.3], [0.6, 0.0], [-0.4, 1.0]])
        uv = cam.project(xz)
        self.assertEqual(tuple(uv[0]), (640.0, 360.0))
        self.assertGreater(uv[1, 0], uv[0, 0])      # +X is image right
        self.assertGreater(uv[1, 1], uv[0, 1])      # lower Z is lower in the image
        np.testing.assert_allclose(cam.unproject(uv), xz)

    def test_framing_puts_ground_and_width(self):
        cam = SideCamera.framing(-0.8, 0.8, 0.0, 1.0)
        self.assertAlmostEqual(cam.project(np.array([0.0, 0.0]))[1], 0.78 * 720)
        span = cam.project(np.array([[-0.8, 0], [0.8, 0]]))
        self.assertAlmostEqual(span[1, 0] - span[0, 0], 0.5 * 1280)
        self.assertAlmostEqual(cam.ortho_scale, 1280 / cam.px_per_m)

    def test_scaled_keeps_world_mapping(self):
        cam = SideCamera(1280, 720, 400.0, 0.0, 0.2).scaled(0.5)
        self.assertEqual((cam.width, cam.height, cam.px_per_m), (640, 360, 200.0))

    def test_json_round_trip(self):
        cam = SideCamera(640, 360, 123.5, 0.2, -0.1)
        self.assertEqual(SideCamera.from_json(cam.to_json()), cam)
