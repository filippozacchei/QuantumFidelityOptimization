# discard all warnings (stderr)
include("bayes_opt.jl")
redirect_stderr(devnull)
using .RealBO, Random

include("calibration.jl")
include("caleb.jl")
t = 100.0
res = ideal(t)                # [fid, f_cl, f_sb, A, delta_phi]
f_cl0, f_sb0, A0 = res[2], res[3], res[4]

function scaled_Q(x)
    f_cl = f_cl0 - 2e4 + x[1]*4e4
    f_sb = f_sb0 - 2e4 + x[2]*4e4
    Q_val = Q1(t, f_cl, f_sb, A0)
    return clamp(Q_val, 0.0, 1.0)
end

scaled_bounds = [(0.0,1.0), (0.0,1.0)]
X, y, x_best, y_best = RealBO.bayesopt(
    scaled_Q;
    bounds=scaled_bounds,
    n_init=50,
    n_iter=100,
    M=2000,
    q=1,
    xi=0.01,
    maximize=true,
    seed=42,
    giffile="bo_fidelity_2.gif",
    fps=2,
)
