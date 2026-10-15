"""Orthographic side camera on Blender -Y looking +Y: image right is world +X, image up is world +Z.

Used by system Python and by Blender's Python, so it needs only numpy and the standard library."""
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SideCamera:
    width: int
    height: int
    px_per_m: float
    center_x: float
    center_z: float

    @property
    def ortho_scale(self):
        return max(self.width, self.height) / self.px_per_m

    def project(self, xz):
        xz = np.asarray(xz, dtype=float)
        u = self.width / 2 + (xz[..., 0] - self.center_x) * self.px_per_m
        v = self.height / 2 - (xz[..., 1] - self.center_z) * self.px_per_m
        return np.stack([u, v], axis=-1)

    def unproject(self, uv):
        uv = np.asarray(uv, dtype=float)
        x = (uv[..., 0] - self.width / 2) / self.px_per_m + self.center_x
        z = (self.height / 2 - uv[..., 1]) / self.px_per_m + self.center_z
        return np.stack([x, z], axis=-1)

    def scaled(self, factor):
        return SideCamera(round(self.width * factor), round(self.height * factor), self.px_per_m * factor,
                          self.center_x, self.center_z)

    def to_json(self):
        return asdict(self)

    @classmethod
    def from_json(cls, data):
        return cls(int(data["width"]), int(data["height"]), float(data["px_per_m"]), float(data["center_x"]),
                   float(data["center_z"]))

    @classmethod
    def framing(cls, min_x, max_x, min_z, max_z, width=1280, height=720, fill=0.5, ground_v=0.78):
        """The body spans `fill` of the width and the ground (z = 0) sits at `ground_v` of the height."""
        px_per_m = fill * width / (max_x - min_x)
        if max_z * px_per_m > ground_v * height * 0.95:
            px_per_m = ground_v * height * 0.95 / max_z
        center_z = (ground_v * height - height / 2) / px_per_m
        return cls(width, height, px_per_m, (min_x + max_x) / 2, center_z)
