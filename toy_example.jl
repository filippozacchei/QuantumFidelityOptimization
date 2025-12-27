# toy_example.jl
# Assumes you already defined:
#   - module RealBO with: bayesopt(...) returning BOResult
#   - module RealBOPlots with: plot2d(res; ...) and animate2d(res; ...)

using .BayesOpt
using .BayesOptPlots
using Plots

# Toy objective: maximum at (0.3, -0.2), value = 1
f(x) = 1.0 - (x[1] - 0.3)^2 - (x[2] + 0.2)^2

bounds = [(-1.0, 1.0), (-1.0, 1.0)]

res = BayesOpt.bayesopt(f;
    bounds=bounds,
    n_init=6,
    n_iter=25,
    xi=0.01,
    maximize=true,
    seed=1
)

# Best found (from stored evaluations)
best_idx = argmax(res.y)
println("Best x = ", res.X[best_idx], "   best y = ", res.y[best_idx])

# Plot diagnostics at iteration t=10 (after init)
display(BayesOptPlots.plot2d(res; t=10, f_true=f))

# Animate the whole run
anim, fps = BayesOptPlots.animate2d(res; f_true=f, nx=50, ny=50, fps=10)
gif(anim, "toy_bo.gif", fps=fps)
