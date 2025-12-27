module BayesOptPlots
using Plots
using Main.BayesOpt

function plot2d(res::BayesOpt.BOResult; t::Int=size(res.X,2)-res.n_init,
                nx::Int=40, ny::Int=40, f_true::Union{Nothing,Function}=nothing)

    @assert length(res.bounds) == 2 "plot2d only supports d=2"

    n  = res.n_init + t
    Xn = res.X[:, 1:n]          # d×n
    y  = res.y[1:n]

    lb = [b[1] for b in res.bounds]
    ub = [b[2] for b in res.bounds]
    xs = range(lb[1], ub[1], length=nx)
    ys = range(lb[2], ub[2], length=ny)

    gp, yμ, yσ = BayesOpt.fit_gp(Xn, y)

    Zμ   = zeros(ny, nx)
    Zσ   = zeros(ny, nx)
    Zerr = f_true === nothing ? nothing : zeros(ny, nx)

    for (iy,yv) in enumerate(ys), (ix,xv) in enumerate(xs)
        μ, σ2 = BayesOpt.gp_predict_f1(gp, [xv, yv])
        m = yμ + yσ * μ
        s = sqrt(max(σ2, 0.0)) * yσ
        Zμ[iy,ix] = m
        Zσ[iy,ix] = s
        if Zerr !== nothing
            Zerr[iy,ix] = abs(m - f_true([xv, yv]))
        end
    end

    p_mean = contourf(xs, ys, Zμ; title="Posterior mean", xlabel="x₁", ylabel="x₂")
    scatter!(p_mean, Xn[1,:], Xn[2,:]; ms=3, color=:white, label="samples")

    p_unc = contourf(xs, ys, Zσ; title="Uncertainty σ(x)", xlabel="x₁", ylabel="x₂")
    scatter!(p_unc, Xn[1,:], Xn[2,:]; ms=3, color=:white, label=false)

    best_trace = accumulate(max, y)
    p_trace = plot(best_trace; title="Best-so-far", xlabel="eval", ylabel="best f(x)", lw=2)

    if Zerr === nothing
        return plot(p_mean, p_unc, p_trace; layout=(1,3), size=(1200,400))
    else
        p_err = contourf(xs, ys, Zerr; title="|μ(x) − f(x)|", xlabel="x₁", ylabel="x₂")
        scatter!(p_err, Xn[1,:], Xn[2,:]; ms=3, color=:white, label=false)
        return plot(p_mean, p_unc, p_err, p_trace; layout=(2,2), size=(1000,800))
    end
end

function animate2d(res::BayesOpt.BOResult; f_true::Union{Nothing,Function}=nothing,
                   nx::Int=40, ny::Int=40, fps::Int=10)
    anim = @animate for t in 1:(size(res.X,2) - res.n_init)
        plot2d(res; t=t, nx=nx, ny=ny, f_true=f_true)
    end
    return anim, fps
end

end # module
