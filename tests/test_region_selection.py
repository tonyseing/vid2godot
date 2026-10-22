"""Production candidate decisions over real projected rectangle/limb triangle masks."""
import ast
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy import ndimage

from anim import fitconstraints, raster, silhouette_regions
from anim.camera import SideCamera


SOURCE = Path(__file__).resolve().parents[1] / 'anim' / 'blender' / 'fit_clip.py'


class TriangleLevel:
    """Exact triangle mask oracle for small real geometry, not prescribed scores."""
    def __init__(self, masks, faces):
        self.masks = [mask.ravel() for mask in masks]
        self.h, self.w = masks[0].shape
        self.faces = faces

    def occupancy(self, uv):
        return raster.rasterize(uv, self.faces, self.w, self.h).ravel(), 0


def geometry_namespace(masks, region_frames):
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    body = np.array([[30, 30], [90, 30], [90, 70], [30, 70]], float)
    limb = np.array([[65, 65], [75, 65], [75, 90], [65, 90]], float)

    def projected(x):
        # Root translation moves both meshes; the distal coordinate changes only the visible limb.
        points = np.concatenate([body, limb + [x[2], 0]]) + [x[0], x[1]]
        return points * x[3], None

    ns = {'np': np, 'math': math, 'fitconstraints': fitconstraints, 'silhouette_regions': silhouette_regions,
          'raster': raster, 'faces': faces,
          'SideCamera': SideCamera, 'W_MARKS': .5, 'LEVELS': (1.0,), 'LEVEL_STEPS': {1.0: (1.0,)},
          'SWEEPS': 4, 'MAX_REPEAT': 6, 'SCALE_STEP': .0025, 'TIE': 1e-9,
          'N': 3, 'SCALE': 3, 'limits': [(-20, 20)], 'MOVES': [(0, None), (1, None), (2, None)],
          'base_camera': SideCamera(128, 128, 1, 0, 0), 'PIVOT': (0, 0), 'work_scale': 1,
          'region_frames': region_frames, 'region_cache': {}, 'usable_observations': {}, 'observation_frames': {},
          'brief': {'iouThreshold': .9}, 'posed_uv': projected, 'constraint_search': {},
          'levels': {1.0: TriangleLevel(masks, faces)}}
    names = {'candidate_key', 'prefer_candidate', 'descend', 'shared_camera', 'delta_for', 'clamp',
             'region_metrics', 'region_key'}
    tree = ast.parse(SOURCE.read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), ns)

    def objective(x, f, prior, k, w_marks=.5):
        candidate = ns['levels'][1.0].occupancy(projected(x)[0])[0].reshape(128, 128)
        source = masks[f]
        return 40 * (1 - raster.iou(source, candidate)) + silhouette_regions.evaluate(
            source, candidate, region_frames.get(f, []), 1)['loss']

    ns['score'] = objective
    ns['observation_residuals'] = lambda x, f: np.empty((0, 2))
    return ns, projected, faces


def sampled_geometry_namespace():
    """Real facing-filtered sampler omits a flap that the final all-face raster must include."""
    faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
    square = np.array([[8.25, 8.25], [17.75, 8.25], [17.75, 17.75], [8.25, 17.75]])
    flap = np.array([[10.25, 10.25], [10.25, 14.75], [14.75, 10.25]])
    rest_uv = np.concatenate([square, flap])
    source = raster.rasterize(rest_uv.astype(np.float32), faces, 32, 32)
    regions = {0: [{'id': 'unchanged-body', 'bounds': [8, 8, 18, 18], 'visibility': 'visible',
                    'minIoU': .85, 'weight': 1}]}

    def projected(x):
        # Rest flap is inside the square; moving it right exposes its unsampled pixels.
        return np.concatenate([square, flap + [x[2], 0]]), None

    ns = dict(np=np, math=math, raster=raster, silhouette_regions=silhouette_regions,
              fitconstraints=fitconstraints, faces=faces, rest_uv=rest_uv,
              width=32, height=32, frame_count=1, masks=[source],
              outside=[ndimage.distance_transform_edt(~source)],
              region_frames=regions, work_scale=1., region_cache={}, usable_observations={},
              brief={'iouThreshold': .9}, posed_uv=projected,
              paw_bones=[], observation_frames={}, marks={}, joint_marks={}, PERIODIC={})
    functions = {'barycentric_grid', 'samples_for', 'erode', 'inside_distance', 'score',
                 'paw_term', 'observation_term', 'region_metrics', 'region_key',
                 'candidate_key', 'prefer_candidate'}
    assignments = {'SPACING', 'W_IOU', 'W_CHAMFER', 'W_MARKS', 'OUT_OF_FRAME',
                   'MAX_BOUNDARY', 'MAX_VIDEO_POINTS', 'tri', 'edge', 'signed', 'facing', 'area'}
    definitions = []
    for node in ast.parse(SOURCE.read_text()).body:
        if (isinstance(node, ast.FunctionDef) and node.name in functions or
                isinstance(node, ast.ClassDef) and node.name == 'Level' or
                isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in assignments
                    for group in node.targets for target in ast.walk(group))):
            definitions.append(node)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), ns)
    ns['levels'] = {1.: ns['Level'](1.)}
    return ns, source, faces


class RegionSelectionTests(unittest.TestCase):
    def test_score_first_feasibility_uses_final_all_triangle_mask_not_facing_samples(self):
        ns, source, faces = sampled_geometry_namespace()
        feasible = np.array([0., 0., 0., 1.])
        false_feasible = np.array([0., 0., 10., 1.])
        threshold = ns['brief']['iouThreshold']
        level = ns['levels'][1.]

        # Real production scoring runs first, just as it does in prefer_candidate.
        loss = ns['score'](false_feasible, 0, None, 1.)
        uv, _ = ns['posed_uv'](false_feasible)
        sampled, lost = level.occupancy(uv)
        sampled = sampled.reshape(source.shape)
        exact = raster.rasterize(uv.astype(np.float32), faces, level.w, level.h)
        regions = ns['region_frames'][0]
        exact_iou = raster.iou(source, exact)
        exact_local = silhouette_regions.evaluate(source, exact, regions, ns['work_scale'])
        self.assertEqual(lost, 0)
        self.assertGreaterEqual(raster.iou(source, sampled), threshold)
        self.assertTrue(silhouette_regions.evaluate(source, sampled, regions, ns['work_scale'])['pass'])
        self.assertLess(exact_iou, threshold)
        self.assertTrue(exact_local['pass'], 'the visible unchanged-body crop is still locally feasible')

        # Exercise the independent saved-float32 projection path used by the final body gate.
        with TemporaryDirectory() as directory:
            projection_path = Path(directory) / 'projection.npz'
            masks_path = Path(directory) / 'masks.npz'
            np.savez(projection_path, uv=uv.astype(np.float32)[None], faces=faces)
            np.savez(masks_path, masks=source[None])
            final = raster.score_fit({'frames': [{'index': 0}]}, projection_path, masks_path,
                                     threshold, {})
        self.assertEqual(final['lowFrames'], [0])
        self.assertEqual(final['frames'][0]['iou'], round(exact_iou, 4))

        expected = (exact_iou, exact_local['pass'])
        self.assertEqual(ns['region_metrics'](false_feasible, 0), expected,
                         'sampled score must not certify a body rejected by the full-triangle gate')
        key = ns['region_key'](false_feasible, 0, loss)
        self.assertTrue(key[0], 'the sampled false pass must be ranked body-infeasible')
        self.assertAlmostEqual(key[1], threshold - exact_iou)
        self.assertEqual(key[2], not exact_local['pass'])

        # Also cover the opposite order: score cannot poison an already exact cached result.
        ns['region_cache'].clear()
        self.assertEqual(ns['region_metrics'](false_feasible, 0), expected)
        ns['score'](false_feasible, 0, None, 1.)
        self.assertEqual(ns['region_metrics'](false_feasible, 0), expected)
        self.assertEqual(ns['region_key'](false_feasible, 0, loss), key)

        # Identical sampled objectives must not trap the selector at the falsely feasible incumbent.
        feasible_loss = ns['score'](feasible, 0, None, 1.)
        self.assertEqual(feasible_loss, loss)
        feasible_uv, _ = ns['posed_uv'](feasible)
        self.assertGreaterEqual(raster.iou(source, raster.rasterize(
            feasible_uv.astype(np.float32), faces, level.w, level.h)), threshold)
        self.assertLess(ns['candidate_key'](feasible, 0, feasible_loss),
                        ns['candidate_key'](false_feasible, 0, loss))
        self.assertTrue(ns['prefer_candidate'](feasible, false_feasible, 0, None, 1.))
        self.assertFalse(ns['prefer_candidate'](false_feasible, feasible, 0, None, 1.))

    def source(self, scale=1):
        body = np.array([[30, 30], [90, 30], [90, 70], [30, 70]], float)
        limb = np.array([[65, 65], [75, 65], [75, 90], [65, 90]], float)
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
        return raster.rasterize(np.concatenate([body, limb]) * scale, faces, 128, 128)

    def regions(self):
        return {0: [{'id': 'visible-limb', 'bounds': [60, 70, 90, 95], 'visibility': 'visible',
                     'minIoU': .85, 'weight': 1}]}

    def test_body_feasible_incumbent_beats_lower_soft_loss_body_failure(self):
        ns, _, _ = geometry_namespace([self.source()], self.regions())
        incumbent = np.array([0., 0., 5., 1.])
        tempting = np.array([15., 0., -15., 1.])
        self.assertLess(ns['score'](tempting, 0, None, 1), ns['score'](incumbent, 0, None, 1))
        self.assertLess(ns['candidate_key'](incumbent, 0, ns['score'](incumbent, 0, None, 1)),
                        ns['candidate_key'](tempting, 0, ns['score'](tempting, 0, None, 1)))
        self.assertFalse(ns['prefer_candidate'](tempting, incumbent, 0, None, 1))

    def test_initial_bad_body_can_improve_before_local_refinement(self):
        ns, _, _ = geometry_namespace([self.source()], self.regions())
        far = np.array([30., 0., 0., 1.])
        closer = np.array([20., 0., 0., 1.])
        self.assertLess(ns['candidate_key'](closer, 0, ns['score'](closer, 0, None, 1)),
                        ns['candidate_key'](far, 0, ns['score'](far, 0, None, 1)))

    def test_descend_cannot_buy_local_improvement_by_losing_a_feasible_body(self):
        source = self.source()
        ns, _, _ = geometry_namespace([source], self.regions())
        pose, _ = ns['descend'](np.array([0., 0., 5., 1.]), 0, None, only={0})
        mask = ns['levels'][1.0].occupancy(ns['posed_uv'](pose)[0])[0].reshape(128, 128)
        self.assertGreaterEqual(raster.iou(source, mask), .9)

    def test_original_source_disks_still_rank_before_body_and_region_quality(self):
        ns, _, _ = geometry_namespace([self.source()], self.regions())
        ns['usable_observations'] = {0: [{}]}
        ns['observation_residuals'] = lambda x, f: np.array([[x[0] - 15, 0]])
        body = np.array([0., 0., 0., 1.])
        joint = np.array([15., 0., -15., 1.])
        self.assertTrue(ns['prefer_candidate'](joint, body, 0, None, 1))

    def test_twice_rounded_pyramid_keeps_valid_source_edge_crop_scoreable(self):
        ns, _, _ = geometry_namespace([self.source()], {})
        tree = ast.parse(SOURCE.read_text())
        definitions = [node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == 'Level' or
                       isinstance(node, ast.FunctionDef) and node.name == 'erode']
        ns.update(MAX_BOUNDARY=800, W_IOU=40., W_CHAMFER=1., OUT_OF_FRAME=50.,
                  work_scale=.5, region_frames={0: [{'id': 'edge', 'bounds': [80, 20, 101, 80],
                  'visibility': 'visible', 'minIoU': .85, 'weight': 1}]})
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), ns)
        level = ns['Level'].__new__(ns['Level'])
        level.k, level.h, level.w = .25, 16, 12
        level.masks = [np.ones(16 * 12, bool)]
        level.areas, level.dist = [16 * 12], [np.zeros((16, 12), float)]
        level.points, level.points2 = [np.empty((0, 2))], [np.empty(0)]
        a, b = np.meshgrid(np.arange(129), np.arange(129), indexing='ij')
        keep = a + b <= 128
        grid = np.stack([1 - (a[keep] + b[keep]) / 128, a[keep] / 128, b[keep] / 128], axis=1)
        level.groups = [(np.array([[0, 1, 2], [0, 2, 3]]).T, grid)]
        values = level.loss(np.array([[0, 0], [50, 0], [50, 64], [0, 64]], float), 0, include_regions=True)
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_full_coverage_body_feasibility_does_not_depend_on_score_cache_order(self):
        source = np.ones((128, 128), bool)
        ns, _, _ = geometry_namespace([source], self.regions())
        tree = ast.parse(SOURCE.read_text())
        functions = {'erode', 'score', 'paw_term', 'observation_term'}
        definitions = [node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == 'Level' or
                       isinstance(node, ast.FunctionDef) and node.name in functions]
        ns.update(MAX_BOUNDARY=800, W_IOU=40., W_CHAMFER=1., OUT_OF_FRAME=50.,
                  paw_bones=[], marks={}, joint_marks={}, PERIODIC={})
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), ns)
        level = ns['Level'].__new__(ns['Level'])
        level.k, level.h, level.w = 1., 128, 128
        level.masks, level.areas = [source.ravel()], [source.size]
        level.dist, level.points, level.points2 = [np.zeros(source.shape)], [np.empty((0, 2))], [np.empty(0)]
        a, b = np.meshgrid(np.arange(129), np.arange(129), indexing='ij')
        keep = a + b <= 128
        grid = np.stack([1 - (a[keep] + b[keep]) / 128, a[keep] / 128, b[keep] / 128], axis=1)
        level.groups = [(np.array([[0, 1, 2], [0, 2, 3]]).T, grid)]
        uv = np.array([[0, 0], [127.999, 0], [127.999, 127.999], [0, 127.999]], float)
        occupied, lost = level.occupancy(uv)
        self.assertTrue(occupied.all())
        self.assertEqual(lost, 0)
        ns['levels'], ns['posed_uv'] = {1.: level}, lambda pose: (uv, None)
        ns['faces'] = np.array([[0, 1, 2], [0, 2, 3]])
        pose = np.array([0., 0., 0., 1.])
        cold = ns['candidate_key'](pose, 0, 0.)
        self.assertFalse(cold[0], 'actual full coverage is body-feasible')
        ns['score'](pose, 0, None, 1.)
        self.assertEqual(ns['candidate_key'](pose, 0, 0.), cold,
                         'legacy no-outline loss sentinel must not overwrite geometric body feasibility')

    def test_region_only_shared_camera_keeps_known_joint_body_and_local_feasibility(self):
        masks = [self.source(.9)] * 3
        ns, _, _ = geometry_namespace(masks, self.regions())
        poses = {0: np.array([0., 0., 0., .9]), 1: np.array([0., 0., 0., 1.]),
                 2: np.array([0., 0., 0., 1.])}
        ns['shared_camera'](poses, [0, 1, 2])
        self.assertEqual(len({pose[3] for pose in poses.values()}), 1)
        for f, pose in poses.items():
            candidate = ns['levels'][1.0].occupancy(ns['posed_uv'](pose)[0])[0].reshape(128, 128)
            self.assertGreaterEqual(raster.iou(masks[f], candidate), .9)
        actual = ns['levels'][1.0].occupancy(ns['posed_uv'](poses[0])[0])[0].reshape(128, 128)
        self.assertTrue(silhouette_regions.evaluate(masks[0], actual, self.regions()[0], 1)['pass'])


if __name__ == '__main__':
    unittest.main()
