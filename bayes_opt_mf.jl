################ MULTI-FIDELITY BO (KOH) WITH PROPER INPUT SCALING ###########
module RealBO_MF

using Random
using LinearAlgebra
using Distributions
using Plots
using GaussianProcesses
using Statistics   

# ================================================================
# Utilities
# ================================================================
scale_to_unit(x::AbstractVector, lb::AbstractVector, ub::AbstractVector) =
    (x .- lb) ./ (ub .- lb)

unscale_from_unit(z::AbstractVector, lb::AbstractVector, ub::AbstractVector) =
    lb .+ z .* (ub .- lb)

# check point is new
function is_new_point(x, X; tol=1e-9)
    for xi in X
        if maximum(abs.(x .- xi)) ≤ tol
            return false
        end
    end
    return true
end

# ================================================================
# Fit GP (on *scaled* inputs!)
# ================================================================
function fit_gp!(X::Vector{Vector{Float64}}, y::Vector{Float64}; noise_floor=1e-5)
    n = length(X)
    d = length(X[1])

    # --- Standardize outputs ---
    yμ = mean(y)
    yσ = max(std(y), 1e-12)
    ystd = (y .- yμ) ./ yσ

    # Put inputs into d×n form
    Xmat = reduce(hcat, X)'   # n × d
    Xgp  = Xmat'              # d × n

    # GP model
    ℓ0  = fill(0.3, d)
    σf0 = 1.0
    σn0 = max(noise_floor, 1e-5)

    gp = GP(Xgp, ystd, MeanZero(), Matern(5/2, ℓ0, σf0), σn0)

    ℓ0  = fill(0.3, d)   # one lengthscale per input dimension
    σf0 = 1.0            # signal std

    kern = SEArd(ℓ0, σf0)
    # gp   = GP(Xgp, ystd, MeanZero(), kern, σn0)

    optimize!(gp)

    return gp, yμ, yσ
end

function gp_predict_f(gp, x_scaled::Vector{Float64})
    μ, σ2 = predict_f(gp, reshape(x_scaled, :, 1))
    return μ[1], σ2[1]
end

# ================================================================
# Expected Improvement
# ================================================================
function ei(gp, x_scaled, fbest_std; xi=0.0)
    μ, σ2 = gp_predict_f(gp, x_scaled)
    Δ = μ - (fbest_std + xi)
    if σ2 ≤ 1e-18
        return max(Δ, 0.0)
    end
    σ = sqrt(σ2)
    γ = Δ / σ
    return Δ * cdf(Normal(),γ) + σ * pdf(Normal(),γ)
end

# ================================================================
# KOH TWO-FIDELITY SURROGATE: f_H = ρ f_L + δ
# ================================================================
function fit_koh!(X_L, y_L, X_H, y_H; noise_floor_lf=1e-1, noise_floor_hf=1e-5)

    # 1) LF surrogate
    gp_L, μ_L, σ_L = fit_gp!(X_L, y_L; noise_floor=noise_floor_lf)

    # 2) predicted LF at HF locations (physical scale)
    μ_L_H = similar(y_H)
    for (i, x_scaled) in enumerate(X_H)
        μstd, _ = gp_predict_f(gp_L, x_scaled)
        μ_L_H[i] = μ_L + σ_L * μstd
    end

    # 3) find ρ
    denom = sum(μ_L_H.^2)
    ρ = denom < 1e-12 ? 1.0 : sum(μ_L_H .* y_H) / denom

    # 4) residuals
    r_H = y_H .- ρ .* μ_L_H

    # 5) GP on residuals
    gp_δ, μ_δ, σ_δ = fit_gp!(X_H, r_H; noise_floor=noise_floor_hf)

    return gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ
end

function predict_koh(
    gp_L, μ_L, σ_L,
    gp_δ, μ_δ, σ_δ,
    ρ,
    x_scaled,
)
    # LF part
    μL_std, σL2_std = gp_predict_f(gp_L, x_scaled)
    μL = μ_L + σ_L * μL_std
    σL = sqrt(max(σL2_std,0)) * σ_L

    # residual part
    μδ_std, σδ2_std = gp_predict_f(gp_δ, x_scaled)
    μδ = μ_δ + σ_δ * μδ_std
    σδ = sqrt(max(σδ2_std,0)) * σ_δ

    μH  = ρ * μL + μδ
    σH2 = (ρ^2)*σL^2 + σδ^2

    return μH, σH2
end

function ei_koh(gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ,
                x_scaled, fbest_H; xi=0.0)
    μH, σH2 = predict_koh(gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ, x_scaled)
    Δ = μH - (fbest_H + xi)
    if σH2 ≤ 1e-18
        return max(Δ,0.0)
    end
    σH = sqrt(σH2)
    γ  = Δ/σH
    return Δ*cdf(Normal(),γ) + σH*pdf(Normal(),γ)
end

# --------------------------------------------------------------------
#  Optional visualisation for 2D case (f_cl, f_sb)
# --------------------------------------------------------------------
function koh_visualization(
    gp_L, μ_L, σ_L,
    gp_δ, μ_δ, σ_δ,
    ρ,
    xs, ys,
    X_L, X_H,
    y_H,        # <-- add this
    Ztrue;
    it = 0
)
    nx = length(xs)
    ny = length(ys)

    Zμ  = zeros(ny, nx)
    Zσ  = zeros(ny, nx)
    Zerr = zeros(ny, nx)

    for (iy, yv) in enumerate(ys), (ix, xv) in enumerate(xs)
        x_scaled = [xv, yv]  # already scaled
        μH, σH2 = predict_koh(gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ, x_scaled)
        Zμ[iy, ix]   = μH
        Zσ[iy, ix]   = sqrt(max(σH2,0))
        Zerr[iy, ix] = abs(Zμ[iy, ix] - Ztrue[iy, ix])
    end

    # ---- HF posterior mean ----
    p1 = contourf(xs, ys, Zμ;
        title = "Posterior HF mean (iter $it)",
        xlabel="scaled f_cl", ylabel="scaled f_sb",
        c=:plasma
    )
    scatter!(p1, [x[1] for x in X_L], [x[2] for x in X_L];
        m=:xcross, c=:white, ms=4, label="LF")
    scatter!(p1, [x[1] for x in X_H], [x[2] for x in X_H];
        m=:circle, c=:black, ms=4, label="HF")

    # ---- Uncertainty ----
    p2 = contourf(xs, ys, Zσ;
        title = "Posterior σ (iter $it)",
        xlabel="scaled f_cl", ylabel="scaled f_sb",
        c=:viridis
    )
    scatter!(p2, [x[1] for x in X_L], [x[2] for x in X_L];
        m=:xcross, c=:white, ms=4, label="LF")
    scatter!(p2, [x[1] for x in X_H], [x[2] for x in X_H];
        m=:circle, c=:black, ms=4, label="HF")

    # ---- Error ----
    p3 = contourf(xs, ys, Zerr;
        title="|μ_H - Q_H|",
        xlabel="scaled f_cl", ylabel="scaled f_sb",
        c=:magma
    )

    # ---- Best HF trace ----
    best_trace = accumulate(max, y_H)
    p4 = plot(best_trace;
        xlabel="HF eval index", ylabel="best HF",
        title="Best HF-so-far",
        lw=2, c=:orange
    )

    return plot(p1, p2, p3, p4; layout=(2,2), size=(1000,800))
end

# ================================================================
# MULTI-FIDELITY BO (WITH SCALING)
# ================================================================
function mf_bayesopt(
    Q_fun, bounds, N_L, N_H;
    n_init_L=8, n_init_H=3,
    n_iter=40,
    M=2000,
    xi=0.01,
    seed=nothing,
    giffile=nothing,
    fps=2,
)

    if seed !== nothing
        Random.seed!(seed)
    end

    d = length(bounds)
    lb = [b[1] for b in bounds]
    ub = [b[2] for b in bounds]

    # --- define experiment functions ---
    f_L(x_phys) = Q_fun(x_phys, N_L)
    f_H(x_phys) = Q_fun(x_phys, N_H)

    # --- storage (scaled inputs) ---
    X_L = Vector{Vector{Float64}}()
    y_L = Float64[]
    X_H = Vector{Vector{Float64}}()
    y_H = Float64[]

    # ============================================================
    # INITIAL DESIGNS
    # ============================================================
    for _ in 1:n_init_L
        x_phys = [rand()*(ub[j]-lb[j]) + lb[j] for j in 1:d]
        x_scaled = scale_to_unit(x_phys, lb, ub)
        push!(X_L, x_scaled)
        push!(y_L, f_L(x_phys))
    end

    for _ in 1:n_init_H
        x_phys = [rand()*(ub[j]-lb[j]) + lb[j] for j in 1:d]
        x_scaled = scale_to_unit(x_phys, lb, ub)
        push!(X_H, x_scaled)
        push!(y_H, f_H(x_phys))
    end

    # ============================================================
    # One iteration of MF-BO
    # ============================================================
    function bo_step(it, X_L, y_L, X_H, y_H)
        println("MF-BO iter $(it) |LF|=$(length(X_L)) |HF|=$(length(X_H))")

        # fit surrogate
        gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ =
            fit_koh!(X_L, y_L, X_H, y_H)

        fbest_H = maximum(y_H)

        # --- generate candidate set (physical) ---
        cand_phys = [[rand()*(ub[j]-lb[j]) + lb[j] for j in 1:d] for _ in 1:M]
        cand_scaled = [scale_to_unit(x, lb, ub) for x in cand_phys]

        # remove duplicates in scaled GP input space
        cand_scaled = [x for x in cand_scaled if is_new_point(x, vcat(X_L, X_H))]

        if isempty(cand_scaled)
            x_phys = [rand()*(ub[j]-lb[j])+lb[j] for j in 1:d]
            cand_phys = [x_phys]
            cand_scaled = [scale_to_unit(x_phys, lb, ub)]
        end

        # evaluate EI and variance
        EI_H  = zeros(length(cand_scaled))
        Var_H = zeros(length(cand_scaled))
        for i in 1:length(cand_scaled)
            μH, σH2 = predict_koh(gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ, cand_scaled[i])
            EI_H[i]  = ei_koh(gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ,
                               cand_scaled[i], fbest_H; xi=xi)
            Var_H[i] = σH2
        end

        # choose next LF/HF evaluation
        idx_H = argmax(EI_H)
        idx_L = argmax(Var_H)

        gain_H = EI_H[idx_H]/N_H
        gain_L = Var_H[idx_L]/N_L

        if gain_H ≥ gain_L
            # evaluate HIGH fidelity
            x_phys = cand_phys[idx_H]
            x_scaled = cand_scaled[idx_H]
            y_new = f_H(x_phys)
            push!(X_H, x_scaled)
            push!(y_H, y_new)
            println(" → HF at ", x_phys, " = ", y_new)
        else
            # evaluate LOW fidelity
            x_phys = cand_phys[idx_L]
            x_scaled = cand_scaled[idx_L]
            y_new = f_L(x_phys)
            push!(X_L, x_scaled)
            push!(y_L, y_new)
            println(" → LF at ", x_phys, " = ", y_new)
        end


        return X_L, y_L, X_H, y_H
    end

    # ============================================================
    # RUN LOOP
    # ============================================================
    if giffile === nothing
        # ================== NO GIF ==================
        for it in 1:n_iter
            X_L, y_L, X_H, y_H = bo_step(it, X_L, y_L, X_H, y_H)
        end

    else
        # ================== GIF MODE ==================
        println("→ Creating GIF: $giffile")

        anim = Animation()

        # Precompute scaled grid (for posterior plots)
        nx, ny = 50, 50
        xs = range(0, 1, length=nx)
        ys = range(0, 1, length=ny)

        # Precompute true HF landscape on scaled grid
        Ztrue = zeros(ny, nx)
        for (iy, yv) in enumerate(ys), (ix, xv) in enumerate(xs)
            x_phys = unscale_from_unit([xv, yv], lb, ub)
            Ztrue[iy, ix] = f_H(x_phys)
        end

        for it in 1:n_iter
            X_L, y_L, X_H, y_H = bo_step(it, X_L, y_L, X_H, y_H)

            # Fit surrogate for visualisation
            gp_L, μ_L, σ_L, gp_δ, μ_δ, σ_δ, ρ =
                fit_koh!(X_L, y_L, X_H, y_H)

            fig = koh_visualization(
                gp_L, μ_L, σ_L,
                gp_δ, μ_δ, σ_δ,
                ρ,
                xs, ys,
                X_L, X_H,
                y_H,       # <-- pass y_H here
                Ztrue;
                it=it
            )

            frame(anim, fig)
        end

        gif(anim, giffile, fps=fps)
        println("GIF saved to $giffile")
    end

    # best HF
    best_idx = argmax(y_H)
    x_best_scaled = X_H[best_idx]
    x_best_phys = unscale_from_unit(x_best_scaled, lb, ub)
    y_best_H = y_H[best_idx]

    return X_L, y_L, X_H, y_H, x_best_phys, y_best_H
end

end # module
