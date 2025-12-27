using .BayesOpt
using .BayesOptPlots
using Plots
using Random, Distributions

σ = 0.01
f_true(x) = 1 - (x[1]-0.3)^2 - (x[2]+0.2)^2
f_noisy(x) = f_true(x) + rand(Normal(0, σ))

bounds = [(-1.0, 1.0), (-1.0, 1.0)]

res, x_rec, y_rec = BayesOpt.bayesopt(f_noisy;
    bounds=bounds,
    n_init=6,
    n_iter=100,
    xi=0.01,
    maximize=true,
    seed=1,
    obs_noise=σ
)


best_idx = argmax(res.y)
x_best = vec(res.X[:, best_idx])
println("Best observed x = ", x_best, "   best observed y = ", res.y[best_idx])
println("Recommended (GP-mean) x = ", x_rec, "   recommended y ≈ ", y_rec)

display(BayesOptPlots.plot2d(res; f_true=f_true))

anim, fps = BayesOptPlots.animate2d(res; f_true=f_true, nx=50, ny=50, fps=5)
gif(anim, "toy_bo.gif", fps=fps)
