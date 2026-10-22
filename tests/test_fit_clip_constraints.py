"""Exercise the Blender fitter's numeric integration without importing Blender."""
import ast
import math
from pathlib import Path
import unittest

import numpy as np

from anim import fitconstraints
from anim.camera import SideCamera


SOURCE = Path(__file__).resolve().parents[1] / "anim" / "blender" / "fit_clip.py"


def fitter_namespace(**values):
    tree = ast.parse(SOURCE.read_text())
    # Compile the production numeric functions, not a reimplementation of their decisions.
    names = {"camera_for", "observation_residuals", "candidate_key", "prefer_candidate",
             "delta_for", "clamp", "descend", "swapped", "better", "fixed_neighbours", "relabel",
             "shared_camera"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"np": np, "math": math, "fitconstraints": fitconstraints,
                 "W_MARKS": 0.5, "LEVELS": (1.0,), "LEVEL_STEPS": {1.0: (0.5, 0.25)},
                 "SWEEPS": 4, "MAX_REPEAT": 6, "SCALE_STEP": 0.0025,
                 "TIE": 1e-9, "W_SWAP_MARKS": 1.0, "W_SWAP_ACCEL": 400.0,
                 "constraint_search": {}, "region_frames": {}, "region_cache": {}, "SideCamera": SideCamera}
    namespace.update(values)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace, tree


class PlanarRig:
    """Two genuinely articulated rigid segments, using the fitter's x/z coordinates."""
    def points_xz(self, angles, dx, dz, wanted):
        points = []
        for bone, end in wanted:
            angle = angles[0 if bone == "left" else 1]
            head = np.array([dx, dz])
            points.append(head if end == "head" else head + [math.cos(angle), math.sin(angle)])
        return np.array(points)


class FitClipConstraintTests(unittest.TestCase):
    def geometry(self, observations):
        namespace, tree = fitter_namespace(
            FAST=True, rig=PlanarRig(), N=4, SCALE=4, limits=[(-1.5, 1.5), (-1.5, 1.5)],
            MOVES=[(i, None) for i in range(4)], base_camera=SideCamera(200, 200, 100.0, 0.0, 0.0),
            PIVOT=(100.0, 100.0), work_scale=1.0, usable_observations=observations,
            observation_frames=observations)
        return namespace, tree

    def test_shared_camera_preserves_feasible_rigid_segment(self):
        namespace, tree = self.geometry({})
        pose = np.array([0.0, 0.0, 0.3, -0.3, 0.9])
        targets = namespace["camera_for"](0.9).project(
            namespace["rig"].points_xz(pose[2:4], 0.0, 0.0, [("left", "head"), ("left", "tail")]))
        observations = {0: [{"bone": "left", "end": end, "point": target, "uncertaintyPx": 0.1}
                            for end, target in zip(("head", "tail"), targets)]}
        namespace["usable_observations"] = namespace["observation_frames"] = observations
        namespace["score"] = lambda x, f, prior, k, w_marks=0.5: (
            float(np.square(namespace["observation_residuals"](x, f)).sum()) + (x[4] - 1.0) ** 2)
        xs = {0: pose.copy(), 1: pose.copy(), 2: pose.copy()}
        xs[1][4] = xs[2][4] = 1.0
        namespace.update(xs=xs, frames=[0, 1, 2], scale_free=True)
        camera_selection = next(node for node in tree.body
                                if isinstance(node, ast.If) and isinstance(node.test, ast.Name)
                                and node.test.id == "scale_free")
        exec(compile(ast.Module(body=[camera_selection], type_ignores=[]), str(SOURCE), "exec"), namespace)
        self.assertEqual(len({x[4] for x in xs.values()}), 1, "output must have one camera")
        residuals = namespace["observation_residuals"](xs[0], 0)
        self.assertTrue(np.all(np.linalg.norm(residuals, axis=1) <= 1.0 + fitconstraints.TOL),
                        "median camera destroys the observed rigid segment's known feasible scale")
        self.assertEqual(namespace["cam_scale"], xs[0][4])

    def test_relabel_new_feasibility_outranks_acceleration(self):
        namespace, _ = self.geometry({})
        pose = np.array([0.0, 0.0, 0.6, -0.6, 1.0])
        pair = ("L", "R", [2], [3])
        corrected = namespace["swapped"](pose, pair)
        target = namespace["camera_for"](1.0).project(namespace["rig"].points_xz(
            corrected[2:4], 0.0, 0.0, [("left", "tail")]))[0]
        original = namespace["camera_for"](1.0).project(namespace["rig"].points_xz(
            pose[2:4], 0.0, 0.0, [("left", "tail")]))[0]
        observations = {1: [{"bone": "left", "end": "tail", "point": target,
                             "uncertaintyPx": float(np.linalg.norm(original - target) / 2.0)}]}
        namespace.update(usable_observations=observations, observation_frames=observations,
                         pairs=[pair], bones=["left", "right"], marks={}, joint_marks={},
                         project={"paws": {"L": "left", "R": "right"},
                                  "chains": {"L": ["left"], "R": ["right"]}})
        namespace["observation_term"] = lambda x, f, cam, selected: float(
            np.square(namespace["observation_residuals"](x, f)).sum())
        xs = {f: pose.copy() for f in range(3)}
        self.assertEqual(namespace["relabel"](xs, [1]), 1)
        np.testing.assert_array_equal(xs[1], corrected)
        np.testing.assert_array_equal(xs[0], pose)
        np.testing.assert_array_equal(xs[2], pose)

    def short_sequence_relabel(self, count):
        namespace, _ = self.geometry({})
        pose = np.array([0.0, 0.0, 0.6, -0.6, 1.0])
        pair = ("L", "R", [2], [3])
        corrected = namespace["swapped"](pose, pair)
        points = [namespace["camera_for"](1.0).project(namespace["rig"].points_xz(
            value[2:4], 0.0, 0.0, [("left", "tail")]))[0] for value in (pose, corrected)]
        observations = {0: [{"bone": "left", "end": "tail", "point": points[1],
                             "uncertaintyPx": float(np.linalg.norm(points[0] - points[1]) / 2.0)}]}
        namespace.update(usable_observations=observations, observation_frames=observations,
                         pairs=[pair], bones=["left", "right"], marks={}, joint_marks={},
                         project={"paws": {"L": "left", "R": "right"},
                                  "chains": {"L": ["left"], "R": ["right"]}})
        namespace["observation_term"] = lambda x, f, cam, selected: float(
            np.square(namespace["observation_residuals"](x, f)).sum())
        xs = {f: pose.copy() for f in range(count)}
        self.assertEqual(namespace["relabel"](xs, list(xs)), 1)
        np.testing.assert_array_equal(xs[0], corrected)
        for f in range(1, count):
            np.testing.assert_array_equal(xs[f], pose)

    def test_single_observed_frame_relabels_without_acceleration(self):
        self.short_sequence_relabel(1)

    def test_two_frame_clip_relabels_observed_frame_without_acceleration(self):
        self.short_sequence_relabel(2)

    def test_short_clip_without_observations_keeps_legacy_identity_despite_marks(self):
        for count in (1, 2):
            with self.subTest(frames=count):
                namespace, _ = self.geometry({})
                pose = np.array([0.0, 0.0, 0.6, -0.6, 1.0])
                pair = ("L", "R", [2], [3])
                target = namespace["camera_for"](1.0).project(namespace["rig"].points_xz(
                    namespace["swapped"](pose, pair)[2:4], 0.0, 0.0, [("left", "tail")]))[0]
                namespace.update(pairs=[pair], bones=["left", "right"], marks={0: {"left": target}},
                                 joint_marks={}, project={"paws": {"L": "left", "R": "right"},
                                                         "chains": {"L": ["left"], "R": ["right"]}})
                namespace["mark_errors"] = lambda x, f, cam, paw_names, joint_names: [
                    float(np.linalg.norm(cam.project(namespace["rig"].points_xz(
                        x[2:4], x[0], x[1], [("left", "tail")]))[0] - target))]
                xs = {f: pose.copy() for f in range(count)}
                self.assertEqual(namespace["relabel"](xs, list(xs)), 0)
                for value in xs.values():
                    np.testing.assert_array_equal(value, pose)


if __name__ == "__main__":
    unittest.main()
