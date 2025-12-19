# =========================
# exploratory_mf.jl
# =========================
using Random, Statistics
using Plots                     # for quick heatmaps/contours
redirect_stderr(devnull)

# Assumes: ideal(t), Q(t, f_cl, f_sb, A, N) already defined in scope
# (i.e., include the file where you've defined them before running this)
include("caleb.jl")
include("calibration.jl")

# ---------- Setup ----------
Random.seed!(42)
t = 100.0
res = ideal(t)                              # [fid, f_cl, f_sb, A, delta_phi]
f_cl0, f_sb0, A0 = res[2], res[3], res[4]

# ---------- 1) Fidelity vs N at nominal (f_cl0, f_sb0, A0) ----------
Ns = [1, 5, 10, 20, 50, 100, 200, 500]         # tune as you like
R1  = 25                                      # repeats to estimate mean & std

function fidelity_stats_at_nominal(N; R=R1)
    vals = [Q(t, f_cl0, f_sb0, A0, N) for _ in 1:R]
    μ = mean(vals); s = std(vals); sem = s/sqrt(R)
    return μ, s, sem
end

println("\n=== Fidelity vs N at nominal parameters ===")
println(rpad("N",8), rpad("mean",12), rpad("std",12), "sem")
for N in Ns
    μ, s, sem = fidelity_stats_at_nominal(N)
    println("N = $N, mean = $μ, std = $s, sem = $sem")
end

# Optional: quick line plot
means = Float64[]; sems = Float64[]
for N in Ns
    μ, s, sem = fidelity_stats_at_nominal(N)
    push!(means, μ); push!(sems, sem)
end
pN = plot(Ns, means; xlabel="N", ylabel="Fidelity",
          ribbon=sems, label="mean ± SEM", lw=2,
          title="Fidelity vs N at (f_cl0, f_sb0, A0)")
savefig(pN, "fidelity_vs_N.png")

# ---------- 2) Q-landscape over (f_cl, f_sb) for various N ----------
println("\n=== Q-landscape ===")

# # Define a symmetric window around the nominal values.
Δf1 = 2e3
Δf2 = 2e3
Δf  = 2e3
ΔA  = 8e4

nx, ny, na = 25, 25, 25                        # start small; increase later if needed
f_cl_grid = range(f_cl0 - Δf1, f_cl0 + Δf1, length=nx)
f_sb_grid = range(f_sb0 - Δf2, f_sb0 + Δf2, length=ny)
A_grid = range(A0 - ΔA, A0 + ΔA, length=na)

function landscape(N)
    Z = Array{Float64}(undef, ny, nx)
    for (iy, A) in enumerate(A_grid), (ix, fcl) in enumerate(f_cl_grid)
        println(A)
        Z[iy, ix] = clamp(Q(t, fcl, f_sb0, A, N), 0.0, 1.0)
    end
    return Z
end

Ns_land = [1, 10, 100, 1000, 10000]             # representative fidelities
plots = Vector{Any}(undef, length(Ns_land))
for (k, N) in enumerate(Ns_land)
    println(N)
    landscape
    Z = landscape(N)
    # Use contourf (continuous) or heatmap (fast); choose one.
    p = contourf(f_cl_grid, A_grid, Z;
                 xlabel="f_cl (Hz)", ylabel="A",
                 title="Fidelity landscape, N = $(N)",
                 colorbar_title="F")
    plots[k] = p
    savefig(p, "landscape_A$(N).png")
end

# Combine into one figure for quick comparison
fig = plot(plots...; layout=(1, length(Ns_land)), size=(1000*length(Ns_land), 300))
savefig(fig, "landscape_comparison.png")

println("\nSaved figures:")
println("  - fidelity_vs_N.png")
for N in Ns_land
    println("  - landscape_N$(N).png")
end
println("  - landscape_comparison.png")

# ---------- 3) Optional: 1D slices (f_cl or f_sb) ----------
# Useful if 2D grids are too slow initially.
function slice_over_fcl(N; ny=101)
    fcls = range(f_cl0 - Δf, f_cl0 + Δf, length=ny)
    vals = [Q(t, fcl, f_sb0, A0, N) for fcl in fcls]
    return fcls, vals
end

function slice_over_fsb(N; ny=101)
    fsbs = range(f_sb0 - Δf, f_sb0 + Δf, length=ny)
    vals = [Q(t, f_cl0, fsb, A0, N) for fsb in fsbs]
    return fsbs, vals
end

function slice_over_A(N; ny=101)
    fsbs = range(A0 - ΔA, A0 + ΔA, length=ny)
    vals = [Q(t, f_cl0, f_sb0, AA, N) for AA in fsbs]
    return fsbs, vals
end

fcls, v10 = slice_over_fcl(10)
_,    v100 = slice_over_fcl(100)
_,    v1000 = slice_over_fcl(1000)
_,    v10000 = slice_over_fcl(10000)

p_slice = plot(fcls, v10; label="N=10", xlabel="f_cl (Hz)", ylabel="Fidelity",
               title="1D slice over f_cl at f_sb0, A0")
plot!(p_slice, fcls, v100; label="N=100")
plot!(p_slice, fcls, v1000; label="N=1000")
plot!(p_slice, fcls, v10000; label="N=10000")
savefig(p_slice, "slice_fcl.png")
println("  - slice_fcl.png")

# Compute slices for each fidelity level
fsbs, w10   = slice_over_A(10)
_,    w100  = slice_over_A(100)
_,    w1000 = slice_over_A(1000)
_,    w10000 = slice_over_A(10000)

# Plot and save
p_slice_fsb = plot(fsbs, w10; label="N=10", xlabel="A", ylabel="Fidelity",
                   title="1D slice over A at f_cl0, f_sb0")
plot!(p_slice_fsb, fsbs, w100; label="N=100")
plot!(p_slice_fsb, fsbs, w1000; label="N=1000")
plot!(p_slice_fsb, fsbs, w10000; label="N=10000")

savefig(p_slice_fsb, "slice_A.png")
println("  - slice_A.png")
