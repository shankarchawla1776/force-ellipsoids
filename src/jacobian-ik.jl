using LinearAlgebra
using StaticArrays
using Plots
function forward_kinematics(q, L)
    n = length(q)
    x, y = 0.0, 0.0
    theta = 0.0
    for i in 1:n
        theta += q[i]
        x += L[i] * cos(theta)
        y += L[i] * sin(theta)
    end
    return [x, y]
end

# q
function compute_joint_positions(q, L)
    n = length(q)
    positions = Vector{SVector{2,Float64}}(undef, n + 1)
    positions[1] = SVector(0.0, 0.0)
    theta = 0.0
    x, y = 0.0, 0.0
    for i in 1:n
        theta += q[i]
        x += L[i] * cos(theta)
        y += L[i] * sin(theta)
        positions[i+1] = SVector(x, y)
    end
    return positions
end

# Jacobian in 2xn space
function jacobian_planar(q, L)
    n = length(q)
    J = zeros(2, n)
    for i in 1:n
        dxi = 0.0
        dyi = 0.0
        for j in i:n
            θ_j = sum(q[1:j])
            dxi -= L[j] * sin(θ_j)
            dyi += L[j] * cos(θ_j)
        end
        J[1, i] = dxi
        J[2, i] = dyi
    end
    return J
end


n = 3                          # n DOFs
L = [1.0, 0.8, 0.6]            # link lengths (meters)
q = [pi / 6, pi / 4, -pi / 8]         # joint angles (radians)

nominalW = ones(n)             # [1.0, 1.0, 1.0]
W = Diagonal(nominalW)         # Tr(W) = n

x_ee = forward_kinematics(q, L)
positions = compute_joint_positions(q, L)
J = jacobian_planar(q, L)

# force ellipsoids are computed as
#   E(q) = J(q) * W⁻¹ * J(q)ᵀ, F(q) = inv(E(q)).
W_inv = Diagonal(1.0 ./ nominalW)
E = J * W_inv * J'
F_mat = inv(E)

# find eigenvectors (prinpical directions) and values (FP potentials)
eigF = eigen(F_mat)
eigvalsF = eigF.values
eigvecsF = eigF.vectors
# The axis = sqrt(eigenvals)
sqrt_eig = sqrt.(eigvalsF)

# some angle t
ts = range(0, stop=2π, length=100)
ellipse_points = [eigvecsF * [sqrt_eig[1] * cos(t); sqrt_eig[2] * sin(t)] for t in ts]
# center wellipse on effector
ellipse_points = [x_ee .+ p for p in ellipse_points]
x_ellipse = [p[1] for p in ellipse_points]
y_ellipse = [p[2] for p in ellipse_points]
# Λ (α) = diag(λ₁ (α) λ₂ (α))
# 𝐱 = Q(α) Λ (α) ⋅ [sin(θ) cos(θ)]^⊤
# find axis
if eigvalsF[1] >= eigvalsF[2]
    major_length = sqrt_eig[1]
    minor_length = sqrt_eig[2]

    major_vector = eigvecsF[:, 1]
    minor_vector = eigvecsF[:, 2]
else
    major_length = sqrt_eig[2]
    minor_length = sqrt_eig[1]

    major_vector = eigvecsF[:, 2]
    minor_vector = eigvecsF[:, 1]
end

rx = [p[1] for p in positions]
ry = [p[2] for p in positions]

plt_static = plot(rx, ry, marker=:circle, lw=2, label="Robot Links",
    xlabel="x", ylabel="y", aspect_ratio=1, title="Planar Robot with Force Ellipsoid")

scatter!(plt_static, rx, ry, ms=5, label="Joints")
plot!(plt_static, x_ellipse, y_ellipse, lw=2, lc=:red, label="Force Ellipsoid")

quiver!([x_ee[1]], [x_ee[2]],
    quiver=([major_length * major_vector[1]], [major_length * major_vector[2]]),
    arrow=:closed, color=:green, label="Major Axis")

quiver!([x_ee[1]], [x_ee[2]],
    quiver=([minor_length * minor_vector[1]], [minor_length * minor_vector[2]]),
    arrow=:closed, color=:blue, label="Minor Axis")

savefig(plt_static, "force_ellipsoid.png")

for i in 1:n
    all_x_points = Float64[]
    all_y_points = Float64[]

    plt_iter = plot(legend=false, xlabel="x", ylabel="y", aspect_ratio=1,
        title="Force Ellipsoid Variation for Weight $(i)")
    plot!(plt_iter, rx, ry, marker=:circle, lw=2)
    scatter!(plt_iter, rx, ry, ms=5)

    # 100 times
    for iter in 0:99
        # s(iter+1)/100.
        new_weight = (iter + 1) / 100
        newW = ones(n)           # set Tr(W) = n
        newW[i] = new_weight     #
        W_iter = Diagonal(newW)
        W_inv_iter = Diagonal(1.0 ./ newW)
        # ellipsoid
        E_iter = J * W_inv_iter * J'
        F_iter = inv(E_iter)

        # eigensolve
        eigF_iter = eigen(F_iter)
        eigvals_iter = eigF_iter.values
        eigvecs_iter = eigF_iter.vectors
        sqrt_eig_iter = sqrt.(eigvals_iter)

        ellipse_points_iter = [eigvecs_iter * [sqrt_eig_iter[1] * cos(θ); sqrt_eig_iter[2] * sin(θ)] for θ in ts]
        ellipse_points_iter = [x_ee .+ p for p in ellipse_points_iter]
        x_ellipse_iter = [p[1] for p in ellipse_points_iter]
        y_ellipse_iter = [p[2] for p in ellipse_points_iter]

        append!(all_x_points, x_ellipse_iter)
        append!(all_y_points, y_ellipse_iter)

        t_norm = iter / 99
        col = get(cgrad(:viridis), t_norm)
        plot!(plt_iter, x_ellipse_iter, y_ellipse_iter, color=col, lw=1)
    end

    xmin = minimum(all_x_points)
    xmax = maximum(all_x_points)
    ymin = minimum(all_y_points)
    ymax = maximum(all_y_points)

    margin = 0.1 * max(xmax - xmin, ymax - ymin)
    xlims!(plt_iter, xmin - margin, xmax + margin)
    ylims!(plt_iter, ymin - margin, ymax + margin)

    filename = "force_ellipsoid_iter_weight$(i).png"
    savefig(plt_iter, filename)
end
