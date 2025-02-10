#!/usr/bin/env python3
from pathlib import Path
from sys import argv

import pinocchio

pinocchio_model_dir = Path(__file__).parent.parent / "models"

urdf_filename = (
    "example_arm.urdf"
    if len(argv) < 2
    else Path(argv[1])
)

model = pinocchio.buildModelFromUrdf(urdf_filename)
print(mode.name)
data = model.createData()
q = pinocchio.randomConfiguration(model)

pinocchio.forwardKinematics(model, data, q)

for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
