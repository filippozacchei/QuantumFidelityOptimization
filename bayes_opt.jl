module BayesOpt

using Random, Statistics, Distributions, GaussianProcesses

struct BOResult
    X::Matrix{Float64}   # d×n
    y::Vector{Float64}
    bounds::Vector{Tuple{Float64,Float64}}
    n_init::Int
    maximize::Bool
end


# -------- helpers --------
@inline function rand_in_box(lb::Vector{Float64}, ub::Vector{Float64})
    d = length(lb)
    x = Vector{Float64}(undef, d)
    @inbounds for j in 1:d
        x[j] = rand() * (ub[j] - lb[j]) + lb[j]
    end
    return x
end

function fit_gp(X::Matrix{Float64}, y::Vector{Float64}; obs_noise::Union{Nothing,Float64}=nothing)
    # X: d×n
    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ

    d, n = size(X)
    ℓ0  = fill(0.3, d)
    σf0 = 1.0
    σn0 = obs_noise === nothing ? 1e-5 : obs_noise

    gp = GP(X, ystd, MeanZero(), Matern(3/2, ℓ0, σf0), σn0)
    if obs_noise === nothing
        optimize!(gp)              
    else
        optimize!(gp; noise=false) 
    end
    return gp, yμ, yσ
end

@inline function gp_predict_f1(gp, x::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(x, :, 1))
    return μ[1], σ2[1]
end

@inline function ei(gp, x::Vector{Float64}, fbest_std::Float64; xi=0.0)
    μ, σ2 = gp_predict_f1(gp, x)
    Δ = μ - (fbest_std + xi)
    if σ2 ≤ 1e-18
        return max(Δ, 0.0)
    end
    σ = sqrt(σ2)
    γ = Δ / σ
    return Δ * cdf(Normal(), γ) + σ * pdf(Normal(), γ)
end

function bayesopt(f; bounds::Vector{Tuple{Float64,Float64}},
                  n_init::Int=8, n_iter::Int=30, M::Int=2000,
                  xi::Float64=0.01, maximize::Bool=true, seed=nothing,
                  obs_noise::Union{Nothing,Float64}=nothing)

    seed !== nothing && Random.seed!(seed)

    d  = length(bounds)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    f_eval = maximize ? f : (x -> -f(x))

    X = Matrix{Float64}(undef, d, 0)
    y = Float64[]

    for _ in 1:n_init
        x = rand_in_box(lb, ub)
        X = hcat(X, x)
        push!(y, f_eval(x))
    end

    for _ in 1:n_iter
        gp, yμ, yσ = fit_gp(X, y; obs_noise=obs_noise)

        y_std = (y .- yμ) ./ yσ
        fbest_std = maximum(y_std)

        best_x = nothing
        best_a = -Inf
        for _ in 1:M
            x = rand_in_box(lb, ub)
            a = ei(gp, x, fbest_std; xi=xi)
            if a > best_a
                best_a = a
                best_x = x
            end
        end

        X = hcat(X, best_x)
        push!(y, f_eval(best_x))
    end

    # recommendation: argmax posterior mean
    gp, yμ, yσ = fit_gp(X, y; obs_noise=obs_noise)
    best_x = nothing
    best_m = -Inf
    for _ in 1:M
        x = rand_in_box(lb, ub)
        μstd, _ = gp_predict_f1(gp, x)
        μ = yμ + yσ * μstd
        if μ > best_m
            best_m = μ
            best_x = x
        end
    end

    res = BOResult(X, maximize ? y : (-y), bounds, n_init, maximize)
    x_rec = best_x
    y_rec = maximize ? best_m : -best_m
    return res, x_rec, y_rec
end
            
end
