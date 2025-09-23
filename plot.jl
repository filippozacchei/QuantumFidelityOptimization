using Plots
using Measures
include("calibration.jl")

t = 100.0
res = ideal(t)                # [fid, f_cl, f_sb, A, delta_phi]
f_cl0, f_sb0, A0 = res[2], res[3], res[4]

println("Baseline values:")
println(" f_cl0 = $f_cl0")
println(" f_sb0 = $f_sb0")
println(" A0    = $A0")

# Small intervals centered on the baseline values
f_cl_range = range(f_cl0 - 2e4, f_cl0 + 2e4,  length=100)
f_sb_range = range(f_sb0 - 2e4, f_sb0 + 2e4,  length=100)

# Shifted ranges for plotting (relative offsets)
f_cl_ticks = [x - f_cl0 for x in f_cl_range]
f_sb_ticks = [y - f_sb0 for y in f_sb_range]

# Safe fidelity evaluation with print
function safeQ(t, fcl, fsb, A)
    println("Running Q with f_cl=$(round(fcl,digits=1)), f_sb=$(round(fsb,digits=1)), A=$(round(A,digits=1))")
    try
        return Q(t, fcl, fsb, A)
    catch e
        println(" -> ERROR at this point: $e")
        return NaN
    end
end

# Build 2D map
function map2d(xs, ys, f)
    Z = Matrix{Float64}(undef, length(xs), length(ys))
    for (i, x) in enumerate(xs)
        for (j, y) in enumerate(ys)
            println("Grid point ($i,$j) / ($(length(xs)),$(length(ys)))")
            Z[i,j] = f(x,y)
        end
    end
    return Z
end

# Scans
Z_cl_sb = map2d(f_cl_range, f_sb_range,
    (fcl, fsb) -> safeQ(t, fcl, fsb, A0))

# Common color scale
finite_vals = filter(!isnan, vcat(vec(Z_cl_sb)))
cmin, cmax = minimum(finite_vals), maximum(finite_vals)

# Build heatmaps with adjusted ticks and labels
p1 = heatmap(f_cl_ticks, f_sb_ticks, Z_cl_sb',
             xlabel="Δf_cl", ylabel="Δf_sb",
             title="Fidelity vs f_cl & f_sb",
             xtickfont=font(6), ytickfont=font(6),  # smaller ticks
             guidefont=font(8), titlefont=font(9),  # axis labels + title
             left_margin=5mm, bottom_margin=5mm,    # fix label cutoff
             clims=(cmin,cmax), color=:plasma)

note = "Axes are centered on the optimal values:\n" *
       "f_cl0 = $(round(f_cl0, digits=1)), " *
       "f_sb0 = $(round(f_sb0, digits=1)), " *
       "A0 = $(round(A0, digits=1))"

# First row: the 3 heatmaps
row1 = plot(p1)

# Stack them vertically
plt = plot(row1)

savefig(plt, "fidelity_local_scan_offsets.png")
println("Saved -> fidelity_local_scan_offsets.png")

