using .RealBO_MF

include("calibration.jl")
include("caleb.jl")

t = 100.0
res = ideal(t)                # [fid, f_cl, f_sb, A, delta_phi]
f_cl0, f_sb0, A0 = res[2], res[3], res[4]

# 2D objective: x = [f_cl, f_sb], A fixed to A0
Q_fun_2d(x, N) = Q(t, x[1], f_sb0, x[2], N)

# Bounds only on f_cl and f_sb
bounds_2d = [
    (f_cl0 - 2e4, f_cl0 + 2e4),   # f_cl
    (A0 - 8e4, A0 + 8e4),   # f_sb
]

X_L, y_L, X_H, y_H, x_best_H, y_best_H =
    RealBO_MF.mf_bayesopt(
        Q_fun_2d,
        bounds_2d,
        10,            # N_L: low-fidelity time (s)
        1000;          # N_H: high-fidelity time (s)
        n_init_L = 50,
        n_init_H = 10,
        n_iter   = 100,
        M        = 2000,
        xi       = 0.001,
        seed     = 42,
        giffile  = "mf_bo_fcl_fsb.gif",   # <-- GIF will be saved here
        fps      = 2,
    )

println("Best HF point: x_best_H = ", x_best_H, ", y_best_H = ", y_best_H)
println("GIF written to mf_bo_fcl_fsb.gif")
