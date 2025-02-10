using LinearAlgebra
using RigidBodyDynamics
using StaticArrays
using MeshCatMechanisms
using MeshCat  #

function rotation_matrix_from_vectors(a::SVector{3,Float64}, b::SVector{3,Float64})
    a_norm = norm(a)
    b_norm = norm(b)
    if a_norm == 0 || b_norm == 0
        return I(3)
    end
    a_unit = a / a_norm
    b_unit = b / b_norm
    c = dot(a_unit, b_unit)
    if isapprox(c, 1.0; atol=1e-6)
        return I(3)
    elseif isapprox(c, -1.0; atol=1e-6)

        axis = normalize(cross(a_unit, SVector(1.0, 0.0, 0.0)))

        if norm(axis) < 1e-6
            axis = normalize(cross(a_unit, SVector(0.0, 1.0, 0.0)))
        end
        angle = π

    else
        angle = acos(c)
        axis = normalize(cross(a_unit, b_unit))
    end
    # rodrigues formula implementation
    K = @SMatrix [  0.0      -axis[3]   axis[2];
                   axis[3]     0.0      -axis[1];
                  -axis[2]   axis[1]     0.0 ]
    return I(3) + sin(angle)*K + (1-cos(angle))*(K*K)
end

# params
g = -9.81

# (m)
link1_length = 1.0    # ground to first joint
link2_length = 0.8    # first to second joint
link3_length = 0.6    # second to third joint
link4_length = 0.4    # third joint to end effector

# (kg)
m1 = 0.3
m2 = 0.25
m3 = 0.2
m4 = 0.15

# COM positions along the link (from joint)
com1 = link1_length/2 * 0.9
com2 = link2_length/2 * 0.85
com3 = link3_length/2 * 0.9
com4 = link4_length/2 * 0.9

# Moments of inertia (scaled rod model)
I1 = (1/12)*m1*link1_length^2 * 1.2
I2 = (1/12)*m2*link2_length^2 * 1.1
I3 = (1/12)*m3*link3_length^2 * 1.15
I4 = (1/12)*m4*link4_length^2 * 1.1

# revolute joints rotate about the negative y–axis, and motion is in the x-z plane
axis = SVector(0.0, -1.0, 0.0)


world = RigidBody{Float64}("world")
arm = Mechanism(world; gravity=SVector(0.0, 0.0, g))

# from Pkg examples
joint1 = Joint("base_joint", Revolute(axis))
inertia1 = SpatialInertia(frame_after(joint1),
                          com = SVector(com1, 0, 0),
                          moment_about_com = I1 * axis * transpose(axis),
                          mass = m1)
link1 = RigidBody(inertia1)
attach!(arm, world, link1, joint1)

joint2 = Joint("elbow_joint", Revolute(axis))
inertia2 = SpatialInertia(frame_after(joint2),
                          com = SVector(com2, 0, 0),
                          moment_about_com = I2 * axis * transpose(axis),
                          mass = m2)
link2 = RigidBody(inertia2)
joint2_placement = Transform3D(frame_before(joint2), frame_after(joint1), SVector(link1_length, 0.0, 0.0))
attach!(arm, link1, link2, joint2; joint_pose=joint2_placement)

joint3 = Joint("wrist_joint", Revolute(axis))
inertia3 = SpatialInertia(frame_after(joint3),
                          com = SVector(com3, 0, 0),
                          moment_about_com = I3 * axis * transpose(axis),
                          mass = m3)
link3 = RigidBody(inertia3)
joint3_placement = Transform3D(frame_before(joint3), frame_after(joint2), SVector(link2_length, 0.0, 0.0))
attach!(arm, link2, link3, joint3; joint_pose=joint3_placement)

# finger tip
joint4 = Joint("finger_joint", Revolute(axis))
inertia4 = SpatialInertia(frame_after(joint4),
                          com = SVector(com4, 0, 0),
                          moment_about_com = I4 * axis * transpose(axis),
                          mass = m4)
link4 = RigidBody(inertia4)
joint4_placement = Transform3D(frame_before(joint4), frame_after(joint3), SVector(link3_length, 0.0, 0.0))
attach!(arm, link3, link4, joint4; joint_pose=joint4_placement)


state = MechanismState(arm)
deg2rad = π/180
q1 = 120 * deg2rad
q2 = 20  * deg2rad
q3 = 40  * deg2rad
q4 = 40  * deg2rad

set_configuration!(state, joint1, q1)
set_configuration!(state, joint2, q2)
set_configuration!(state, joint3, q3)
set_configuration!(state, joint4, q4)

set_velocity!(state, joint1, 0.0)
set_velocity!(state, joint2, 0.0)
set_velocity!(state, joint3, 0.0)
set_velocity!(state, joint4, 0.0)


mvis = MechanismVisualizer(arm, Skeleton(inertias=false))
# 3-point time vector
ts = [0.0, 0.0, 1000.0]
qs = [copy(configuration(state)), copy(configuration(state)), copy(configuration(state))]
animate(mvis, ts, qs; realtimerate=1.0)


W = Diagonal([1.0, 2.0, 3.0, 4.0])
println("Weight matrix W:")
println(W)

q_desired = SVector(q1, 30 * deg2rad, q3, q4)
println(q_desired)

q_current = configuration(state)  #
println(q_current)

q_error = q_current - q_desired
println(q_error)

tau = W * q_error                   # joint torques
println(tau)

# jacobian at the end effector and the spatial jacobian for link-4
J = jacobian(arm, state, link4, zero(SVector{3,Float64}))
# Jacobian is 6xN
J_linear = J[4:6, :]   # 3x4
println(J_linear)

# use peudoinverve
f = pinv(J_linear') * tau
println("\nComputed endpoint force (f):")
println(f)

T_end = transform(state, link4)
endpoint_pos = T_end.translation

f_norm = norm(f)
if f_norm < 1e-6
else
    f_dir = f / f_norm
    default_dir = SVector(0.0, 0.0, 1.0)
    R_arrow = rotation_matrix_from_vectors(default_dir, f_dir)

    # 4×4 homogeneous transform
    T_arrow = Matrix{Float64}(I, 4, 4)
    T_arrow[1:3, 1:3] = R_arrow
    T_arrow[1:3, 4] = endpoint_pos

    arrow_geom = MeshCat.geometry.Arrow(shaftLength = f_norm * 0.8,
                                          shaftDiameter = 0.01,
                                          headLength = f_norm * 0.2,
                                          headDiameter = 0.02)

    setobject!(mvis.vis, "endpoint_force", arrow_geom)
    settransform!(mvis.vis, "endpoint_force", T_arrow)
end
