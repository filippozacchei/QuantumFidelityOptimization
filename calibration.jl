# merged_ionsim.jl

using IonSim
using QuantumOptics
using Random
const pc = IonSim.PhysicalConstants

# Optional: only needed for the noisy/fit-based estimator
try
    using StatsBase
    using LsqFit
catch
    # You can still run Q_det and ideal without these.
end

@inline function bell_fidelity_phi_plus(ρ::AbstractMatrix{<:Complex})
    ϕ   = angle(ρ[1,4])                 # arg of the SS↔DD coherence
    coh = real(exp(-1im*ϕ) * ρ[1,4])     # align onto the real axis
    return 0.5 * (real(ρ[1,1] + ρ[4,4]) + 2*coh)
end

# Global two-qubit rotation (used by the parity-scan estimator)
function R(θ, φ=0)
    U = [cos(θ/2) -1im*sin(θ/2)*exp(-1im*φ);
         -1im*sin(θ/2)*exp(1im*φ)  cos(θ/2)]
    b = SpinBasis(1//2)
    return Operator(tensor(b,b), kron(U,U))
end

# Convert IonSim reduced density operator to a QuantumOptics.Operator on 2 qubits
function ionSim_compatibility(ρ)
    b = SpinBasis(1//2)
    return Operator(tensor(b,b), ρ.data)
end

proj(state, ρ) = real(expect(state ⊗ dagger(state), ρ))

# Shared setup + ideal evolution
function ideal(t)
    c = 299792458
    ca = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])
    laser1 = Laser(pointing=[(1, 1.), (2, 1.)])
    laser2 = Laser(pointing=[(1, 1.), (2, 1.)])

    chain = LinearChain(ions=[ca, ca],
                        comfrequencies=(x=3e6,y=3e6,z=2.5e5),
                        selectedmodes=(;z=[1]))
    chamber = Chamber(iontrap=chain, B=6e-4, Bhat=(x̂ + ẑ)/√2, lasers=[laser1, laser2])
    mode = zmodes(chamber)[1]
    ν = frequency(mode)
    ϵ = 1/(t*1e-6)

    wavelength_from_transition!(laser1, ca, ("S","D"), chamber)
    detuning!(laser1,  ν + ϵ); polarization!(laser1, x̂); wavevector!(laser1, ẑ); phase!(laser1, 0)

    wavelength_from_transition!(laser2, ca, ("S","D"), chamber)
    detuning!(laser2, -ν - ϵ); polarization!(laser2, x̂); wavevector!(laser2, ẑ); phase!(laser2, 0)

    f_b  = c/laser1.λ + laser1.Δ
    f_r  = c/laser2.λ + laser2.Δ
    f_cl = (f_b + f_r)/2
    f_sb = (f_b - f_r)/2

    η = abs(lambdicke(mode, ca, laser1))
    pi_time  = η/ϵ
    intensity = intensity_from_pitime!(1, pi_time, 1, ("S","D"), chamber)
    intensity_from_pitime!(2, pi_time, 1, ("S","D"), chamber)

    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
    tout = 0:0.01:t
    _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)

    ρ = ptrace(sol[end] ⊗ dagger(sol[end]), [3]).data
    fid = bell_fidelity_phi_plus(ρ)
    return (fid=fid, f_cl=f_cl, f_sb=f_sb, A=intensity, delta_phi=0.0)
end

# Deterministic estimator (your second file's intent)
function Q_det(t, f_cl, f_sb, A)
    ca = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])
    laser1 = Laser(pointing=[(1, 1.), (2, 1.)])
    laser2 = Laser(pointing=[(1, 1.), (2, 1.)])

    chain = LinearChain(ions=[ca, ca],
                        comfrequencies=(x=3e6,y=3e6,z=2.5e5),
                        selectedmodes=(;z=[1]))
    chamber = Chamber(iontrap=chain, B=6e-4, Bhat=(x̂ + ẑ)/√2, lasers=[laser1, laser2])
    mode = zmodes(chamber)[1]

    wavelength!(laser1, pc.c/f_cl); detuning!(laser1,  f_sb); polarization!(laser1, x̂); wavevector!(laser1, ẑ)
    wavelength!(laser2, pc.c/f_cl); detuning!(laser2, -f_sb); polarization!(laser2, x̂); wavevector!(laser2, ẑ)
    intensity!(laser1, A); intensity!(laser2, A)

    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
    tout = 0:0.01:t
    _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)
    ρ = ptrace(sol[end] ⊗ dagger(sol[end]), [3]).data
    return bell_fidelity_phi_plus(ρ)
end

# Noisy estimator with sampling + parity scan + cosine fit (your first file's intent)
function Q_noisy(t, f_cl, f_sb, A; N=100, phase_grid=0:0.1:π)
    @assert @isdefined(StatsBase) && @isdefined(LsqFit) "Q_noisy needs StatsBase and LsqFit."

    ca = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])
    laser1 = Laser(pointing=[(1, 1.), (2, 1.)])
    laser2 = Laser(pointing=[(1, 1.), (2, 1.)])

    chain = LinearChain(ions=[ca, ca],
                        comfrequencies=(x=3e6,y=3e6,z=2.5e5),
                        selectedmodes=(;z=[1]))
    chamber = Chamber(iontrap=chain, B=6e-4, Bhat=(x̂ + ẑ)/√2, lasers=[laser1, laser2])
    mode = zmodes(chamber)[1]

    wavelength!(laser1, pc.c/f_cl); detuning!(laser1,  f_sb); polarization!(laser1, x̂); wavevector!(laser1, ẑ)
    wavelength!(laser2, pc.c/f_cl); detuning!(laser2, -f_sb); polarization!(laser2, x̂); wavevector!(laser2, ẑ)
    intensity!(laser1, A); intensity!(laser2, A)

    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
    tout = 0:0.1:t
    _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)

    # Outcome probabilities in computational basis from projectors
    SS = real(expect(ionprojector(chamber, "S","S"), sol[end]))
    SD = real(expect(ionprojector(chamber, "S","D"), sol[end]))
    DS = real(expect(ionprojector(chamber, "D","S"), sol[end]))
    DD = real(expect(ionprojector(chamber, "D","D"), sol[end]))

    # parity values: SS,DD -> +1 ; SD,DS -> -1
    parval = Dict(1=>+1, 2=>-1, 3=>-1, 4=>+1)
    weights = [SS, SD, DS, DD]

    # Estimate P_odd = Pr(parity=-1) with projection noise
    samples = StatsBase.sample(1:4, StatsBase.Weights(weights), N)
    vals = (parval[s] for s in samples)
    P_odd = count(==( -1), vals) / N

    # Reduced density matrix for parity scan
    ρ_red = ptrace(sol[end] ⊗ dagger(sol[end]), [3])
    ρ = ionSim_compatibility(ρ_red)

    b = SpinBasis(1//2)
    SSs = tensor(spinup(b),   spinup(b))
    SDs = tensor(spinup(b),   spindown(b))
    DSs = tensor(spindown(b), spinup(b))
    DDs = tensor(spindown(b), spindown(b))

    meas = zeros(length(phase_grid))
    for (i, φ) in enumerate(phase_grid)
        ρφ = R(π/2, φ) * ρ * dagger(R(π/2, φ))
        p = [proj(SSs, ρφ), proj(SDs, ρφ), proj(DSs, ρφ), proj(DDs, ρφ)]
        p = max.(p, 0)                      # numerical guard
        p ./= sum(p)

        s = StatsBase.sample(1:4, StatsBase.Weights(p), N)
        v = [parval[x] for x in s]
        meas[i] = sum(v) / N                # parity estimate
    end

    model(φ, p) = p[1] .* cos.(p[2] .* φ .+ p[3]) .+ p[4]
    p0 = [0.8, 1.0, 0.0, 0.0]
    fit = LsqFit.curve_fit(model, collect(phase_grid), meas, p0,
                           lower=[-1.0, -2.0, -Inf, -0.1],
                           upper=[ 1.0,  2.0,  Inf,  0.1])
    C = abs(fit.param[1])

    # Same estimator form you used
    return (1 - P_odd + C)/2
end

# Unified optimizer: choose method = :det or :noisy
function trial(t, f_cl, f_sb, A;
               method::Symbol=:det,
               tolerance=0.005, offset=2e3, step=1e2, max_iter=50)

    qfun = method === :noisy ? (args...)->Q_noisy(args...; N=100) : Q_det

    best_fid = qfun(t, f_cl, f_sb, A)
    best_params = (f_cl, f_sb, A)
    iter = 0
    diff = Inf

    while diff > tolerance && iter < max_iter
        iter += 1
        prev = best_fid

        for fcl in best_params[1]-offset:step:best_params[1]+offset
            fid = qfun(t, fcl, best_params[2], best_params[3])
            if fid > best_fid
                best_fid = fid
                best_params = (fcl, best_params[2], best_params[3])
            end
        end

        for fsb in best_params[2]-offset:step:best_params[2]+offset
            fid = qfun(t, best_params[1], fsb, best_params[3])
            if fid > best_fid
                best_fid = fid
                best_params = (best_params[1], fsb, best_params[3])
            end
        end

        for Aval in best_params[3]-offset:step:best_params[3]+offset
            fid = qfun(t, best_params[1], best_params[2], Aval)
            if fid > best_fid
                best_fid = fid
                best_params = (best_params[1], best_params[2], Aval)
            end
        end

        diff = abs(best_fid - prev)
    end

    return (fid=best_fid, t=t, f_cl=best_params[1], f_sb=best_params[2], A=best_params[3], iters=iter)
end
