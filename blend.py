from typing import Any

import bpy
import numpy as np
from numpy.typing import NDArray


class Circle:
    def __init__(
        self,
        radius: float = 0.0,
        speed: float = 0.0,
        angle_i: float = 0.0,
        angles: NDArray[np.float64] | None = None,
    ) -> None:
        self.radius = radius
        self.speed = speed
        self.angle_i = angle_i
        self._angles: NDArray[np.float64] = np.empty((0,), dtype=np.float64)
        self._local_x: NDArray[np.float64] = np.empty((0,), dtype=np.float64)
        self._local_y: NDArray[np.float64] = np.empty((0,), dtype=np.float64)
        if angles is None:
            self._angles = np.empty((0,), dtype=np.float64)
        else:
            self._angles = angles

    @property
    def local_trajectory(self) -> NDArray[np.float64]:
        return np.stack([self._local_x, self._local_y])

    @property
    def angles(self) -> NDArray[np.float64]:
        return self._angles

    @angles.setter
    def angles(self, value: Any) -> None:
        raise AttributeError(
            "Cannot set angles directly. Set new time array to update."
        )

    @property
    def local_x(self) -> NDArray[np.float64]:
        return self._local_x

    @local_x.setter
    def local_x(self, value: Any) -> None:
        raise AttributeError(
            "Cannot set local_x directly. Set new time array to update."
        )

    @property
    def local_y(self) -> NDArray[np.float64]:
        return self._local_y

    @local_y.setter
    def local_y(self, value: Any) -> None:
        raise AttributeError(
            "Cannot set local_y directly. Set new time array to update."
        )

    def update_arrays(self, time: NDArray[np.float64]) -> None:
        if isinstance(time, np.ndarray) and len(time.shape) == 1:
            self._angles = time * self.speed + self.angle_i
            self._local_x = self.radius * np.cos(self._angles)
            self._local_y = self.radius * np.sin(self._angles)
        else:
            raise ValueError(
                "Time must be set to numpy NDArray of shape (n,)."
            )


class Epicycle:
    def __init__(self) -> None:
        self._circles: list[Circle] = []
        self._time: NDArray[np.float64] = np.empty((0,), dtype=np.float64)
        self._x: NDArray[np.float64] = np.empty((0,), dtype=np.float64)
        self._y: NDArray[np.float64] = np.empty((0,), dtype=np.float64)
        self._period_changed = True
        self._trajectory_changed = True
        self._period: float = 0.0
        self._trajectory: NDArray[np.float64] = np.empty(
            (0,), dtype=np.float64
        )

    @property
    def time(self) -> NDArray[np.float64] | None:
        return self._time

    @time.setter
    def time(self, value: Any) -> None:
        if isinstance(value, np.ndarray) and len(value.shape) == 1:
            self._time = value.astype(np.float64)
            self._x = np.zeros_like(self._time)
            self._y = np.zeros_like(self._time)
            for circle in self._circles:
                circle.update_arrays(self._time)
            self._update_xy()
        else:
            raise ValueError("Time must be set to numpy NDArray or None.")

    @property
    def trajectory(self) -> NDArray[np.float64]:
        if self._trajectory_changed:
            self._trajectory = np.stack(
                [self._x, self._y, np.zeros_like(self._time), np.ones_like(self._time)]
            )
            self._trajectory_changed = False

        return self._trajectory

    @trajectory.setter
    def trajectory(self) -> None:
        raise AttributeError(
            "Cannot set trajectory directly. Set new time array to update."
        )

    def add_circle(
        self, radius: float = 0.0, speed: float = 0.0, angle_i: float = 0.0
    ) -> None:
        self._circles.append(Circle(radius, speed, angle_i, self._time))
        self._period_changed = True
        self._trajectory_changed = True

    def add_circles(
        self, radius: list[float], speed: list[float], angle_i: list[float]
    ) -> None:
        if len(radius) != len(speed) != len(angle_i):
            raise ValueError("Input lists must all have the same shape (n,).")
        for i in range(len(radius)):
            self.add_circle(radius[i], speed[i], angle_i[i])
        self._update_xy()

    def _update_xy(self) -> None:
        self._x = np.zeros_like(self._time)
        self._y = np.zeros_like(self._time)
        for circle in self._circles:
            self._x += circle.local_x
            self._y += circle.local_y
        self._trajectory_changed = True

    @property
    def period(self) -> float:
        if self._period_changed:
            speeds = [circle.speed for circle in self._circles]
            self._period = (
                2 * np.pi / np.lcm.reduce(np.array(speeds, dtype=int))
            )
            self._period_changed = False

        return self._period


import os

context = bpy.context

e = Epicycle()
n = 2
radius = [1, 2]  # n * [1]
speed = [2, 7]  # n * [1]
angle_i = n * [0]
e.add_circles(radius/ np.sum(radius), speed, angle_i) # / np.sum(radius)
e.time = np.linspace(0, 2 * np.pi, int(60 * np.max(speed)))
print(f"Number of points: {e.time.shape[0]}")

objs = [context.scene.objects['Light'], context.scene.objects['Cube']]
with context.temp_override(selected_objects=objs):
    bpy.ops.object.delete()
    
# Orient Camera
context.scene.camera.location = (0.0,0.0,5.5)
context.scene.camera.rotation_euler = (0.0,0.0,0.0)

# Create Material
mat = bpy.data.materials.new(name="MagicCircleMaterial")
mat.use_nodes = True
mat_nodes = mat.node_tree.nodes
mat_links = mat.node_tree.links
mat_nodes.clear()
mat_links.clear()
output = mat_nodes.new("ShaderNodeOutputMaterial")
emission = mat_nodes.new("ShaderNodeEmission")
mat_links.new(emission.outputs["Emission"], output.inputs["Surface"])
emission.inputs["Color"].default_value = (1.0, 0.0, 0.0, 1.0)
emission.inputs["Strength"].default_value = 20.0

# Create curve
cu = bpy.data.curves.new(name="Epicycle", type="CURVE")
cu.dimensions = "2D"
cu.bevel_depth = 0.001

spline = cu.splines.new("NURBS")  # poly type
# spline is created with one point add more to match data
spline.points.add(e.trajectory.shape[1] - 1)
spline.points.foreach_set("co", e.trajectory.flatten("F"))
spline.use_endpoint_u = True
ob = bpy.data.objects.new("Poly", cu)

# Assign it to object
if ob.data.materials:
    # assign to 1st material slot
    ob.data.materials[0] = mat
else:
    # no slots
    ob.data.materials.append(mat)

context.collection.objects.link(ob)
context.view_layer.objects.active = ob
ob.select_set(True)

context.scene.world.color = (0.0,0.0,0.0)
context.scene.world.use_nodes = False

context.scene.eevee.use_bloom = True
context.scene.eevee.bloom_radius = 10.0
context.scene.eevee.bloom_intensity = 0.1

filepath = os.path.join(os.getcwd(),"out.png")

context.scene.render.filepath = filepath
render = bpy.ops.render.render(write_still=True)
