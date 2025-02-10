#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

model_path = "example_robot.urdf"
mesh_dir = "."

# urdf_filename = "solo.urdf"
# urdf_model_path = model_path / "solo_description/robots" / urdf_filename

model, collision_model, visual_model = pin.buildModelsFromUrdf(
    urdf_model_path, mesh_dir, pin.JointModelFreeFlyer()
)

try:
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
except ImportError as err:
    print(err)
    sys.exit(0)

viz.loadViewerModel()

q0 = pin.neutral(model)
viz.display(q0)
viz.displayVisuals(True)

mesh = visual_model.geometryObjects[0].geometry
mesh.buildConvexRepresentation(True)
convex = mesh.convex

if convex is not None:
    # default SE3 placement & shift
    # Create a default SE3 placement and shift it along the x-axis.
    placement = pin.SE3.Identity()
    placement.translation[0] = 2.0
    # use a convex  object
    geometry = pin.GeometryObject("convex", 0, placement, convex)
    geometry.meshColor = np.ones(4)

    geometry.overrideMaterial = True
    geometry.meshMaterial = pin.GeometryPhongMaterial()

    geometry.meshMaterial.meshEmissionColor = np.array([1.0, 0.1, 0.1, 1.0])
    geometry.meshMaterial.meshSpecularColor = np.array([0.1, 1.0, 0.1, 1.0])

    geometry.meshMaterial.meshShininess = 0.8

    visual_model.addGeometryObject(geometry)
    #
    viz.rebuildData()
viz2 = MeshcatVisualizer(model, collision_model, visual_model)

viz2.initViewer(viz.viewer)
viz2.loadViewerModel(rootNodeName="pinocchio2")

q = q0.copy()
q[1] = 1.0
viz2.display(q)

q1 = np.array(
    [0.0, 0.0, 0.235, 0.0, 0.0, 0.0, 1.0, 0.8, -1.6, 0.8, -1.6, -0.8, 1.6, -0.8, 1.6]
)

# Random initial velocity.
v0 = np.random.randn(model.nv) * 2
data = viz.data

pin.forwardKinematics(model, data, q1, v0)
fid = model.getFrameId("HR_FOOT")
viz.display()
viz.drawFrameVelocities(fid=fid)

# Set gravity to zero for simulation.
model.gravity.linear[:] = 0.0
dt = 0.01

def sim_loop():
    tau0 = np.zeros(model.nv)
    qs = [q1]
    vs = [v0]
    nsteps = 100
    for i in range(nsteps):
        q = qs[i]
        v = vs[i]

        a1 = pin.aba(model, data, q, v, tau0)
        vnext = v + dt * a1
        qn = pin.integrate(model, q, dt * vnext)

        qs.append(qn)
        vs.append(vnext)
        viz.display(qn)
        viz.drawFrameVelocities(fid=fid)
    return qs, vs

qs, vs = sim_loop()

fid2 = model.getFrameId("FL_FOOT")

def my_callback(i, *args):
    viz.drawFrameVelocities(fid)
    viz.drawFrameVelocities(fid2)
with viz.create_video_ctx("../leap.mp4"):
    viz.play(qs, dt, callback=my_callback)
