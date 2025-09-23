############### REAL BAYESIAN OPTIMISATION (GaussianProcesses.jl) ###############
module RealBO

using Random
using LinearAlgebra
using Distributions
using Plots
using GaussianProcesses
using Statistics   

# ---------------- Utilities ----------------

# HELPERS
scale_to_unit(x::AbstractVector, lb::AbstractVector, ub::AbstractVector) =
    (x .- lb) ./ (ub .- lb)
unscale_from_unit(z::AbstractVector, lb::AbstractVector, ub::AbstractVector) =
    lb .+ z .* (ub .- lb)

# INITIAL CANDIDATES
function rand_candidates(M::Int, lb::AbstractVector, ub::AbstractVector)
    d = length(lb)
    Z = rand(M, d)                 
    eachrow([unscale_from_unit(Z[i,:], lb, ub) for i in 1:M])
end

# IS X A NEW POINT?? OTHERWISE IT BREAKS :(
function is_new_point(x::AbstractVector, X::Vector{Vector{Float64}}; tol::Float64=1e-9)
    for xi in X
        if maximum(abs.(x .- xi)) ≤ tol
            return false
        end
    end
    return true
end

# GP surrogate 
function fit_gp!(X::Vector{Vector{Float64}}, y::Vector{Float64}; noise_floor=1e-5)
    n = length(X)
    d = length(X[1])

    # Standardise outputs
    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ

    # Pack inputs as d×n
    Xmat = reduce(hcat, X)'  
    Xgp = Xmat'

    ℓ0 = fill(0.3, d)  # initial lengthscales on roughly unit box
    σf0 = 1.0
    σn0 = max(noise_floor, 1e-3) # initial noise

    m0   = MeanZero()
    kern = Matern(3/2, ℓ0, σf0)
    gp   = GP(Xgp, ystd, m0, kern, σn0)

    optimize!(gp)  # Type-II ML for θ = (ℓ, σf, σn)

    return gp, yμ, yσ
end

function gp_predict_f(gp, x::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(x, :, 1))
    return μ[1], σ2[1]
end

# Acquisition (EI) -> choose next point
function ei(gp, x::Vector{Float64}, fbest_std::Float64; xi::Float64=0.0)
    μ, σ2 = gp_predict_f(gp, x)
    Δ = μ - (fbest_std + xi)
    if σ2 ≤ 1e-18
        return max(Δ, 0.0)
    end
    σ = sqrt(σ2); γ = Δ/σ
    return Δ * cdf(Normal(), γ) + σ * pdf(Normal(), γ)
end

# Here comes the magic
function bayesopt(f; bounds::Vector{Tuple{Float64,Float64}},
                  n_init::Int=8, n_iter::Int=30, M::Int=2000, q::Int=1,
                  xi::Float64=0.01, maximize::Bool=true, seed=nothing,
                  giffile::Union{Nothing,String}=nothing, fps::Int=2)

    if seed !== nothing
        Random.seed!(seed)
    end

    d = length(bounds)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    # Objective possibly sign-flipped to perform maximisation
    f_eval = maximize ? f : x -> -f(x)

    # ---- initial design ----
    X = Vector{Vector{Float64}}()
    y = Float64[]

    for _ in 1:n_init
        x = [rand()*(ub[j]-lb[j]) + lb[j] for j in 1:d]
        push!(X, x)
        push!(y, f_eval(x))
    end

    # at setup, only once
    nx, ny = 10, 10
    xs = range(lb[1], ub[1], length=nx)
    ys = range(lb[2], ub[2], length=ny)

    # cache true objective on grid
    Ztrue = [f_eval([x,y]) for y in ys, x in xs]

    # ---- helper: one BO iteration ----
    function bo_step(t, X, y)
        display(t)
        # Fit surrogate
        gp, yμ, yσ = fit_gp!(X, y)
        y_std = (y .- yμ) ./ yσ
        fbest_std = maximum(y_std)

        # Candidate set
        cand = [ [rand()*(ub[j]-lb[j]) + lb[j] for j in 1:d] for _ in 1:M ]
        cand = [x for x in cand if is_new_point(x, X; tol=0.0)]
        isempty(cand) && (cand = [[rand()*(ub[j]-lb[j]) + lb[j] for j in 1:100]])

        # Score EI
        vals = [ei(gp, x, fbest_std; xi=xi) for x in cand]
        order = sortperm(vals, rev=true)

        # Batch: evaluate top-q
        x_next = cand[order[1]]
        y_next = f_eval(x_next)
        push!(X, x_next); push!(y, y_next)

        # -------------- NEW PLOTS if d==2 --------------
        fig = nothing
        if d == 2
            gp, yμ, yσ = fit_gp!(X, y)
            nx, ny = 10, 10
            xs = range(lb[1], ub[1], length=nx)
            ys = range(lb[2], ub[2], length=ny)

            # Posterior mean + uncertainty
            Zμ  = zeros(ny, nx)
            Zσ  = zeros(ny, nx)
            Zerr = zeros(ny, nx)
            for (iy,yv) in enumerate(ys), (ix,xv) in enumerate(xs) 
                μ, σ2 = gp_predict_f(gp, [xv,yv])
                Zμ[iy,ix]  = yμ + yσ*μ
                Zσ[iy,ix]  = sqrt(max(σ2,0.0)) * yσ
                Zerr[iy,ix] = abs(Zμ[iy,ix] - Ztrue[iy,ix])
            end

            p_mean = surface(xs, ys, Zμ; xlabel="x₁", ylabel="x₂", zlabel="μ(x)",
                             title="Posterior mean (iter $t)", c=:plasma)
            scatter!([X[i][1] for i in 1:length(X)],
                     [X[i][2] for i in 1:length(X)],
                     y; marker=:circle, ms=3, color=:black, label="samples")

            p_unc  = contourf(xs, ys, Zσ; xlabel="x₁", ylabel="x₂",
                            title="Uncertainty σ(x)", c=:viridis)
            scatter!(p_unc, [x[1] for x in X], [x[2] for x in X],
                    ms=3, color=:white, label="samples")

            p_err  = contourf(xs, ys, Zerr; xlabel="x₁", ylabel="x₂",
                            title="|μ(x) - f(x)|", c=:magma)
            scatter!(p_err, [x[1] for x in X], [x[2] for x in X],
                    ms=3, color=:white, label="samples")

            best_trace = accumulate(max, y)
            p_trace = plot(best_trace; xlabel="eval", ylabel="best f(x)",
                        lw=2, color=:orange, title="Best-so-far")

            fig = plot(p_mean, p_unc, p_err, p_trace; layout=(2,2),
                    size=(1000,800))
        end
        return X, y, fig
    end

    # ---- run loop ----
    if giffile === nothing
        for t in 1:n_iter
            X, y, fig = bo_step(t, X, y)
            fig !== nothing && display(fig)
        end
    else
        anim = @animate for t in 1:n_iter
            X, y, fig = bo_step(t, X, y)
            fig === nothing ? plot() : fig
        end
        gif(anim, giffile, fps=fps)
    end

    # Final best
    best_idx = argmax(y)
    x_best = X[best_idx]
    y_best = maximize ? y[best_idx] : -y[best_idx]

    return X, (maximize ? y : (-y)), x_best, y_best
end

end # module
