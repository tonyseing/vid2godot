import json
import tempfile
import unittest
from pathlib import Path

from anim.contract import ContractError, check_rig, load_project_contract

PROJECT = json.loads((Path(__file__).parent / "fixtures" / "test_contract.json").read_text())


def rig(**changes):
    bones = [{"name": n, "parent": None, "head": [0, 0, 0], "tail": [0, 0, 1], "alongSide": 0.0}
             for n in PROJECT["requiredBones"]]
    data = {"armature": "rig", "bones": bones, "triangles": 100, "forwardSign": 1,
            "meshes": [{"name": "body", "skinned": True, "vertices": 8, "unweightedVertices": 0}],
            "collapse": [{"bone": "upper_fl", "edges": 0}]}
    data.update(changes)
    return data


class ContractTests(unittest.TestCase):
    def test_good_rig_passes(self):
        self.assertEqual(check_rig(rig(), PROJECT), ([], []))

    def test_no_armature_points_to_3d_skill(self):
        problems, _ = check_rig(rig(armature=None), PROJECT)
        self.assertIn("game-asset-3d-pipeline", problems[0])

    def test_missing_bone(self):
        problems, _ = check_rig(rig(bones=rig()["bones"][1:]), PROJECT)
        self.assertTrue(any("root" in p for p in problems))

    def test_unweighted_vertices(self):
        meshes = [{"name": "body", "skinned": True, "vertices": 8, "unweightedVertices": 3}]
        self.assertTrue(check_rig(rig(meshes=meshes), PROJECT)[0])

    def test_collapse(self):
        problems, _ = check_rig(rig(collapse=[{"bone": "lower_hl", "edges": 12}]), PROJECT)
        self.assertIn("lower_hl", problems[0])

    def test_sideways_bone(self):
        bones = rig()["bones"]
        bones[4]["alongSide"] = 0.99
        self.assertTrue(check_rig(rig(bones=bones), PROJECT)[0])

    def test_unskinned_mesh_is_a_warning(self):
        meshes = rig()["meshes"] + [{"name": "Icosphere", "skinned": False, "vertices": 42, "unweightedVertices": None}]
        problems, warnings = check_rig(rig(meshes=meshes), PROJECT)
        self.assertEqual(problems, [])
        self.assertIn("Icosphere", warnings[0])

    def test_backwards_creature(self):
        self.assertTrue(check_rig(rig(forwardSign=-1), PROJECT)[0])

    def test_project_contract_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text(json.dumps({**PROJECT, "legPairs": [["front_L", "nope"]]}))
            with self.assertRaises(ContractError):
                load_project_contract(path)
            path.write_text(json.dumps({**PROJECT, "forward": "-Z"}))
            with self.assertRaises(ContractError):
                load_project_contract(path)

    def test_malformed_or_missing_contract_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.json"
            path.write_text("{not json")
            with self.assertRaises(ContractError) as caught:
                load_project_contract(path)
            self.assertIn("c.json", str(caught.exception))
            with self.assertRaises(ContractError):
                load_project_contract(Path(tmp) / "absent.json")
