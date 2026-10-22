"""Source-local contour enrollment, actual-projection acceptance and review gates."""
import json
import unittest
from pathlib import Path

import numpy as np

from anim import pipeline, review
from anim.ledger import StageError, sha256_file
from tests import test_articulation


class RegionPipelineTests(unittest.TestCase):
    def setUp(self):
        test_articulation.ArticulationPipelineTests.setUp(self)
        self.ledger.data['stages']['fit']['inputs'].pop('observations')
        (self.work / 'articulation' / 'observations.json').unlink()
        self.regions = self.work / 'silhouette-regions' / 'regions.json'
        self.regions.parent.mkdir()
        video = self.work / 'source.mp4'
        self.document = {'schemaVersion': 1, 'source': {'videoSha256': sha256_file(video),
                         'coordinateSpace': 'registered_source_px'},
                         'frames': {'0': [{'id': 'fold', 'bounds': [20, 20, 42, 42],
                         'visibility': 'visible', 'minIoU': .85, 'weight': 1.0}]}}
        self.regions.write_text(json.dumps(self.document))
        masks = np.zeros((1, 64, 64), bool)
        masks[0, 10:21, 10:21] = True
        np.savez(self.work / 'ingest' / 'masks.npz', masks=masks)
        # A closed pair of triangles exactly covering the source crop in actual projection.
        uv = np.array([[[10, 10], [20, 10], [20, 20], [10, 20]]], float)
        np.savez(self.work / 'fit' / 'projection.npz', uv=uv, faces=[[0, 1, 2], [0, 2, 3]])
        self.ledger.record('ingest', {'video': video},
                           {'meta': self.work / 'ingest' / 'ingest.json', 'masks': self.work / 'ingest' / 'masks.npz'})
        self.ledger.record('fit', {'regions': self.regions, 'marks': self.work / 'fit' / 'marks.json'},
                           {'poses': self.pose_path, 'projection': self.work / 'fit' / 'projection.npz',
                            'report': self.work / 'fit' / 'report.json'})


    def test_actual_projection_not_claimed_pose_metrics_controls_gate(self):
        report = pipeline.assess_regions(self.ledger)
        self.assertEqual(report['status'], 'pass')
        self.assertEqual(report['frames'][0]['regions'][0]['iou'], 1.0)
        self.assertEqual(pipeline.require_articulation(self.ledger)['status'], 'not_assessed')
        projection = self.work / 'fit' / 'projection.npz'
        np.savez(projection, uv=np.array([[[0, 0], [1, 0], [1, 1]]]), faces=[[0, 1, 2]])
        report = pipeline.assess_regions(self.ledger)
        self.assertEqual(report['status'], 'fail')
        report['status'] = 'pass'
        (self.regions.parent / 'assessment.json').write_text(json.dumps(report))
        with self.assertRaises(StageError):
            pipeline.require_articulation(self.ledger)

    def test_local_pass_cannot_trade_away_global_body_quality(self):
        masks = np.ones((1, 64, 64), bool)
        np.savez(self.work / 'ingest' / 'masks.npz', masks=masks)
        self.ledger.record('ingest', {'video': self.work / 'source.mp4'},
                           {'meta': self.work / 'ingest' / 'ingest.json', 'masks': self.work / 'ingest' / 'masks.npz'})
        report = pipeline.assess_regions(self.ledger)
        self.assertEqual(report['frames'][0]['regions'][0]['status'], 'pass')
        self.assertEqual(report['status'], 'fail')
        self.assertFalse(report['bodyGate']['pass'])

    def test_recorded_and_preflight_enrolled_deleted_regions_never_become_legacy(self):
        pipeline.assess_regions(self.ledger)
        self.regions.unlink()
        with self.assertRaises((StageError, OSError)):
            pipeline.require_articulation(self.ledger)
        self.ledger.data['stages']['fit']['inputs'].pop('regions')
        enrollment = self.regions.parent / 'enrollment.json'
        enrollment.write_text(json.dumps({'regions': {'path': str(self.regions.resolve())}}))
        with self.assertRaises((StageError, OSError)):
            pipeline.require_articulation(self.ledger)

    def test_source_hash_and_bounds_refused_before_blender(self):
        from tests.test_pipeline import run
        for change in ('hash', 'bounds'):
            doc = json.loads(json.dumps(self.document))
            if change == 'hash':
                doc['source']['videoSha256'] = '0' * 64
            else:
                doc['frames']['0'][0]['bounds'] = [0, 0, 200, 200]
            self.regions.write_text(json.dumps(doc))
            code, _, error = run('preflight', '--work', str(self.work), '--silhouette-regions',
                                 str(self.regions), '--blender', '/never-run-Blender')
            self.assertEqual(code, 1)
            self.assertNotIn('No such file or directory: \'/never-run-Blender', error)
            self.assertRegex(error.lower(), 'hash|bound|region')

    def test_region_only_preflight_enrolls_before_failed_blender_and_missing_input_blocks_fit(self):
        from tests.test_pipeline import run
        self.ledger.data['stages']['fit']['inputs'].pop('regions')
        self.ledger.path.write_text(json.dumps(self.ledger.data))
        code, _, error = run('preflight', '--work', str(self.work), '--silhouette-regions',
                             str(self.regions), '--blender', '/never-run-Blender')
        self.assertEqual(code, 1)
        self.assertIn('Blender not found', error)
        self.assertFalse((self.work / 'articulation' / 'enrollment.json').exists())
        self.regions.unlink()
        retained = self.work / 'articulation' / 'preflight' / 'regions.json'
        retained.unlink()
        code, _, error = run('fit', '--work', str(self.work), '--blender', '/never-run-Blender')
        self.assertEqual(code, 1)
        self.assertIn('regions.json', error)
        self.assertNotIn('Blender not found', error)

    def test_region_review_cannot_explain_away_failure(self):
        report = {'frames': [{'index': 0, 'iou': .99, 'flags': []}], 'lowFrames': [],
                  'silhouetteRegions': {'status': 'fail', 'frames': []}}
        template = review.make_template('walk', {'poses': self.pose_path}, report, 1)
        filled = {**template, 'reviewer': 'reviewer', 'verdict': 'pass', 'inspectedReference': True,
                  'inspectedOverlays': True, 'inspectedSheet': True, 'observations': 'Looks fine'}
        self.assertTrue(any('region' in p.lower() for p in review.validate(filled, template)))

    def test_unenrolled_legacy_has_no_region_requirement(self):
        self.ledger.data['stages']['fit']['inputs'].pop('regions')
        self.regions.unlink()
        self.assertEqual(pipeline.require_regions(self.ledger)['status'], 'not_assessed')

    def test_assessment_only_enrollment_survives_deleted_source_sidecar(self):
        self.ledger.data['stages']['fit']['inputs'].pop('regions')
        report = pipeline.assess_regions(self.ledger)
        self.assertEqual(report['status'], 'pass')
        self.regions.unlink()
        with self.assertRaises((StageError, OSError)):
            pipeline.require_regions(self.ledger)

    def test_subpixel_width_region_overlay_uses_the_scored_integer_crop(self):
        self.document['frames']['0'][0]['bounds'] = [20, 20, 21, 24]
        self.regions.write_text(json.dumps(self.document))
        report = pipeline.assess_regions(self.ledger)
        output = self.work / 'region-overlays'
        images = review.draw_overlays(self.work, {'frames': [{'index': 0, 'joints': {}}]},
                                      self.work / 'fit' / 'projection.npz',
                                      self.work / 'ingest' / 'masks.npz', output,
                                      region_report=report, work_scale=.5)
        self.assertTrue(images[0].is_file())


if __name__ == '__main__':
    unittest.main()
