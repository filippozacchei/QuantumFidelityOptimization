include("bayes_opt.jl");           using .BayesOpt
include("bayes_hetero_opt.jl");    using .BayesHeteroOpt
include("bayes_opt_plots.jl");     using .BayesPlotting

using Random, Distributions
using Plots

f_true(x) = 1 - (x[1]-0.3)^2 - (x[2]+0.2)^2
f_noisy(x, σ) = f_true(x) + rand(Normal(0, σ))

bounds   = [(-1.0, 1.0), (-1.0, 1.0)]
σ_levels = [0.5, 0.1, 0.05, 0.02, 0.01, 0.005]

res = BayesHeteroOpt.bayesopt_ucb_threshold(f_noisy;
    bounds=bounds,
    σ_levels=σ_levels,
    n_init=6,
    n_iter=100,
    κ=2.0,
    α=0.5,
    seed=1
)

println("Recommended x = ", res.x_rec, "   GP-mean y ≈ ", res.y_rec)

counts = BayesHeteroOpt.count_noise_levels(res)
for σ in res.σ_levels
    println("σ=$(σ): ", counts[σ])
end

display(BayesPlotting.plot2d(res; f_true=f_true))

anim, fps = BayesPlotting.animate2d(res; f_true=f_true, fps=5)
gif(anim, "hetero.gif", fps=fps)
