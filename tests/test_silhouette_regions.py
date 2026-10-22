"""Visible source-contour regions must constrain geometry, not infer hidden joints."""
import copy
import unittest

import numpy as np

from anim import silhouette_regions
from anim.ledger import StageError
from anim.raster import iou


VIDEO_SHA256 = "0b07d00f7130a7d312c58336835abd4ab9710da06acb6beae18091afc3f0e7d4"


def region(bounds=(60, 60, 90, 90), **changes):
    return {"id": "folded-recovery", "bounds": list(bounds), "visibility": "visible",
            "minIoU": .85, "weight": 1.0, **changes}


def document(regions=None):
    return {"schemaVersion": 1,
            "source": {"videoSha256": VIDEO_SHA256, "coordinateSpace": "registered_source_px"},
            "frames": {"85": [region()] if regions is None else regions}}


def folded_masks():
    """Same large torso and paw; wrong middle segment has a different visible fold."""
    source = np.zeros((100, 100), dtype=bool)
    source[5:60, 5:95] = True
    source[60:75, 62:66] = True
    source[71:75, 62:82] = True
    source[71:88, 78:82] = True
    source[84:88, 75:85] = True
    wrong = np.zeros_like(source)
    wrong[5:60, 5:95] = True
    wrong[60:88, 78:82] = True
    wrong[84:88, 75:85] = True
    return source, wrong


_MISSING = object()


class SilhouetteRegionValidationTests(unittest.TestCase):
    def validate(self, doc=_MISSING, **changes):
        args = {"video_sha256": VIDEO_SHA256, "width": 100, "height": 100, "frame_count": 100,
                **changes}
        return silhouette_regions.validate(document() if doc is _MISSING else doc, **args)

    def test_normalizes_frame_indices_without_mutating_source(self):
        doc = document()
        original = copy.deepcopy(doc)
        normalized = self.validate(doc)
        self.assertEqual(list(normalized["frames"]), [85])
        self.assertEqual(normalized["frames"][85], [region()])
        self.assertEqual(doc, original)

    def test_source_schema_and_exact_video_hash_are_required(self):
        invalid = [None, [], {}, {**document(), "schemaVersion": True},
                   {**document(), "schemaVersion": 2}, {**document(), "source": None}]
        for field, value in (("videoSha256", "b" * 64), ("videoSha256", "bad"),
                             ("videoSha256", True), ("coordinateSpace", "working_px")):
            doc = document()
            doc["source"][field] = value
            invalid.append(doc)
        for doc in invalid:
            with self.subTest(doc=doc), self.assertRaises(StageError):
                self.validate(doc)
        for digest in (None, "", "A" * 64, True):
            with self.subTest(expected_digest=digest), self.assertRaises(StageError):
                self.validate(video_sha256=digest)

    def test_positive_integer_source_dimensions_and_frame_count_are_required(self):
        for name in ("width", "height", "frame_count"):
            for value in (0, -1, True, 100.0, float("nan"), None):
                with self.subTest(name=name, value=value), self.assertRaises(StageError):
                    self.validate(**{name: value})

    def test_empty_malformed_and_duplicate_frame_indices_are_rejected(self):
        invalid = [None, [], {}, {"-1": [region()]}, {"100": [region()]}, {"1.5": [region()]},
                   {1.5: [region()]}, {True: [region()]}, {"85": []}, {"85": {}},
                   {"85": [region()], "085": [region()]}]
        for frames in invalid:
            with self.subTest(frames=frames), self.assertRaises(StageError):
                self.validate({**document(), "frames": frames})

    def test_region_identity_visibility_threshold_and_weight_are_validated(self):
        invalid = [None, {}, {**region(), "id": ""}, {**region(), "id": "  "},
                   {**region(), "id": 7}, {**region(), "visibility": "inferred"}]
        for name, values in (("minIoU", (None, True, -1, 0, 1.01, float("nan"), float("inf"))),
                             ("weight", (None, True, 0, -1, float("nan"), float("inf")))):
            invalid.extend({**region(), name: value} for value in values)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(StageError):
                self.validate(document([item]))
        with self.assertRaises(StageError):
            self.validate(document([region(), region()]))
        # The same region identity may recur over different observed frames.
        doc = document()
        doc["frames"]["86"] = [region()]
        self.assertEqual(set(self.validate(doc)["frames"]), {85, 86})

    def test_bounds_are_finite_ordered_and_inside_source_image(self):
        for bounds in (None, [], [0, 1, 2], [0, 1, 2, 3, 4], [True, 0, 2, 2],
                       [0, 0, float("nan"), 2], [0, 0, 2, float("inf")], [-1, 0, 2, 2],
                       [0, -1, 2, 2], [0, 0, 101, 2], [0, 0, 2, 101],
                       [2, 0, 2, 2], [3, 0, 2, 2], [0, 2, 2, 2], [0, 3, 2, 2]):
            with self.subTest(bounds=bounds), self.assertRaises(StageError):
                self.validate(document([{**region(), "bounds": bounds}]))
        self.assertEqual(self.validate(document([region((0, .25, 100, 100))]))["frames"][85][0]["bounds"],
                         [0, .25, 100, 100])


class SilhouetteRegionGeometryTests(unittest.TestCase):
    def test_same_paw_high_global_overlap_does_not_excuse_wrong_visible_fold(self):
        source, wrong = folded_masks()
        self.assertTrue(np.array_equal(source[84:88, 75:85], wrong[84:88, 75:85]))
        self.assertGreater(iou(source, wrong), .95)
        measured = silhouette_regions.evaluate(source, wrong, [region()], 1.0)
        self.assertFalse(measured["pass"])
        self.assertEqual(measured["regions"][0]["status"], "fail")
        expected = iou(source[60:90, 60:90], wrong[60:90, 60:90])
        self.assertAlmostEqual(measured["regions"][0]["iou"], expected)
        self.assertAlmostEqual(measured["loss"], 100 * (1 - expected))

    def test_restoring_visible_contour_passes_without_changing_paw_or_body(self):
        source, wrong = folded_masks()
        restored = wrong.copy()
        restored[60:90, 60:90] = source[60:90, 60:90]
        measured = silhouette_regions.evaluate(source, restored, [region()], 1.0)
        self.assertTrue(measured["pass"])
        self.assertEqual(measured["regions"][0]["status"], "pass")
        self.assertEqual(measured["regions"][0]["iou"], 1.0)
        self.assertEqual(measured["loss"], 0.0)

    def test_source_pixel_bounds_scale_outward_and_use_identical_exclusive_crops(self):
        source = np.zeros((300, 500), dtype=bool)
        fitted = source.copy()
        source[217:263, 365:410] = True
        fitted[217:263, 365:400] = True
        # Outside right and bottom must not leak into either crop.
        fitted[217:263, 410] = True
        fitted[263, 365:410] = True
        item = region((730, 435, 820, 525))
        got = silhouette_regions.evaluate(source, fitted, [item], .5)
        row = got["regions"][0]
        self.assertEqual(row["workBounds"], [365, 217, 410, 263])
        self.assertAlmostEqual(row["iou"], 35 / 45)
        self.assertEqual(row["sourcePixels"], 45 * 46)
        self.assertEqual(row["candidatePixels"], 35 * 46)
        self.assertEqual(row["bounds"], item["bounds"])

    def test_fractional_bounds_and_image_edge_rounding_are_conservative(self):
        source = np.zeros((5, 5), dtype=bool)
        source[0:3, 0:3] = True
        fitted = source.copy()
        fitted[0, 0] = False
        measured = silhouette_regions.evaluate(source, fitted, [region((.5, .5, 5.5, 5.5))], .5)
        self.assertEqual(measured["regions"][0]["workBounds"], [0, 0, 3, 3])
        self.assertAlmostEqual(measured["regions"][0]["iou"], 8 / 9)
        # Source width 11 at scale .5 rounds to working width 6 or 5;
        # outward ceil may exceed the rounded working raster by less than one pixel.
        edge = silhouette_regions.evaluate(source, source, [region((8, 8, 11, 11))], .5)
        self.assertEqual(edge["regions"][0]["workBounds"], [4, 4, 5, 5])
        self.assertFalse(edge["pass"])

    def test_weighted_loss_keeps_each_region_threshold_independent(self):
        source = np.ones((10, 20), dtype=bool)
        fitted = source.copy()
        fitted[:, 10:20] = False
        regions = [region((0, 0, 10, 10), id="body", weight=3),
                   region((10, 0, 20, 10), id="fold", weight=1)]
        measured = silhouette_regions.evaluate(source, fitted, regions)
        self.assertAlmostEqual(measured["loss"], 25.0)
        self.assertFalse(measured["pass"])
        self.assertEqual([r["status"] for r in measured["regions"]], ["pass", "fail"])

    def test_threshold_is_inclusive_and_not_rounded_for_acceptance(self):
        source = np.ones((10, 10), dtype=bool)
        fitted = source.copy()
        fitted.ravel()[:15] = False
        self.assertTrue(silhouette_regions.evaluate(source, fitted, [region((0, 0, 10, 10))])["pass"])
        fitted.ravel()[15] = False
        self.assertFalse(silhouette_regions.evaluate(source, fitted, [region((0, 0, 10, 10))])["pass"])

    def test_hidden_unknown_and_out_of_frame_regions_never_certify_perfect_masks(self):
        source = np.ones((100, 100), dtype=bool)
        for visibility in ("occluded", "unknown", "out_of_frame"):
            doc = document([region(visibility=visibility)])
            normalized = silhouette_regions.validate(doc, VIDEO_SHA256, 100, 100, 100)
            measured = silhouette_regions.evaluate(source, source, normalized["frames"][85])
            with self.subTest(visibility=visibility):
                self.assertFalse(measured["pass"])
                self.assertEqual(measured["regions"][0]["status"], "unresolved")
                self.assertIsNone(measured["regions"][0]["iou"])
                self.assertEqual(measured["loss"], 100.0)

    def test_empty_reference_foreground_is_unresolved_not_perfect_empty_iou(self):
        source = np.zeros((100, 100), dtype=bool)
        for fitted in (source, np.ones_like(source)):
            measured = silhouette_regions.evaluate(source, fitted, [region()])
            self.assertFalse(measured["pass"])
            self.assertEqual(measured["regions"][0]["status"], "unresolved")
            self.assertIsNone(measured["regions"][0]["iou"])
            self.assertEqual(measured["loss"], 100.0)

    def test_empty_candidate_against_visible_source_is_zero_iou_failure(self):
        measured = silhouette_regions.evaluate(np.ones((100, 100), dtype=bool),
                                              np.zeros((100, 100), dtype=bool), [region()])
        self.assertFalse(measured["pass"])
        self.assertEqual(measured["regions"][0]["iou"], 0.0)
        self.assertEqual(measured["regions"][0]["status"], "fail")
        self.assertEqual(measured["loss"], 100.0)

    def test_malformed_masks_scale_or_unvalidated_regions_cannot_certify(self):
        valid = np.ones((100, 100), dtype=bool)
        invalid_masks = [np.ones((100,)), np.ones((100, 100, 1)), np.ones((0, 100)),
                         np.full((100, 100), float("nan")), np.full((100, 100), float("inf")),
                         np.full((100, 100), .5), np.full((100, 100), -1)]
        for invalid in invalid_masks:
            for source, candidate in ((invalid, valid), (valid, invalid)):
                with self.subTest(shape=invalid.shape), self.assertRaises(StageError):
                    silhouette_regions.evaluate(source, candidate, [region()])
        for scale in (0, -1, True, None, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(StageError):
                silhouette_regions.evaluate(valid, valid, [region()], scale)
        invalid_regions = [None, {}, [None], [region(minIoU=float("nan"))],
                           [region(bounds=[0, 0, 2, float("inf")])],
                           [region((0, 0, 101, 100))], [region(), region()]]
        for items in invalid_regions:
            with self.subTest(items=items), self.assertRaises(StageError):
                silhouette_regions.evaluate(valid, valid, items)

    def test_no_regions_has_zero_loss_and_no_acceptance_obligations(self):
        source, fitted = folded_masks()
        self.assertEqual(silhouette_regions.evaluate(source, fitted, []),
                         {"loss": 0.0, "pass": True, "regions": []})
