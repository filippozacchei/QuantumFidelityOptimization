using Revise
using IonSim
using QuantumOptics
using Random
using LsqFit
using StatsBase
const pc = IonSim.PhysicalConstants;

@inline function bell_fidelity_phi_plus(ρ::AbstractMatrix{<:Complex})
    ϕ = angle(ρ[1,4])                 # arg of the SS↔DD coherence
    coh = real(exp(-1im*ϕ)*ρ[1,4])    # align onto the real axis
    return 0.5 * (real(ρ[1,1] + ρ[4,4]) + 2*coh)
end

function R(θ, φ=0)
    R = [cos(θ/2) -1im*sin(θ/2)*exp(-1im*φ); -1im*sin(θ/2)*exp(1im*φ) cos(θ/2)]
    b = SpinBasis(1//2)
    return Operator(tensor(b,b), kron(R,R))
end

function ionSim_compatibility(ρ)
    b = SpinBasis(1//2)
    return Operator(tensor(b,b), ρ.data)
end

function proj(state, ρ)
    return real(expect(state ⊗ dagger(state), ρ))
end

function ideal(t)
    c = 299792458
    ca = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])
    laser1 = Laser(pointing=[(1, 1.), (2, 1.)])
    laser2 = Laser(pointing=[(1, 1.), (2, 1.)])
    chain = LinearChain(
            ions=[ca, ca], comfrequencies=(x=3e6,y=3e6,z=2.5e5), selectedmodes=(;z=[1])
        )
    chamber = Chamber(iontrap=chain, B=6e-4, Bhat=(x̂ + ẑ)/√2, lasers=[laser1, laser2]);
    mode = zmodes(chamber)[1]
    ν = frequency(mode)
    ϵ = 1/(t*1e-6)  # detuning from motional mode

    wavelength_from_transition!(laser1, ca, ("S", "D"), chamber)
    detuning!(laser1, ν + ϵ)
    polarization!(laser1, x̂)
    wavevector!(laser1, ẑ)
    phase!(laser1, 0)

    wavelength_from_transition!(laser2, ca, ("S", "D"), chamber)
    detuning!(laser2, -ν - ϵ)
    polarization!(laser2, x̂)
    wavevector!(laser2, ẑ)
    phase!(laser2, 0)

    f_b = c/laser1.λ+laser1.Δ; f_r = c/laser2.λ+laser2.Δ;
    f_cl = (f_b + f_r) / 2; f_sb = (f_b-f_r)/2;

    η = abs(lambdicke(mode, ca, laser1))
    pi_time = η / ϵ  # setting 'resonance' condition: ηΩ = 1/2ϵ
    intensity = intensity_from_pitime!(1, pi_time, 1, ("S", "D"), chamber)
    intensity_from_pitime!(1, pi_time, 1, ("S", "D"), chamber)
    intensity_from_pitime!(2, pi_time, 1, ("S", "D"), chamber)
    # setup the Hamiltonian
    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf);
    # solve system
    tout = 0:0.01:t
    tout, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)
    ρ = ptrace(sol[end] ⊗ dagger(sol[end]), [3]).data
    fid = bell_fidelity_phi_plus(ρ)

    delta_phi = 0.0
    return [fid, f_cl, f_sb, intensity, delta_phi]
end


# Adds in quantum projection noise and fits contrast to get fidelity
function Q(t, f_cl, f_sb, A, N=100)
    
    ca = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])
    laser1 = Laser(pointing=[(1, 1.), (2, 1.)])
    laser2 = Laser(pointing=[(1, 1.), (2, 1.)])
    chain = LinearChain(
            ions=[ca, ca], comfrequencies=(x=3e6,y=3e6,z=2.5e5), selectedmodes=(;z=[1])
        )
    chamber = Chamber(iontrap=chain, B=6e-4, Bhat=(x̂ + ẑ)/√2, lasers=[laser1, laser2]);
    mode = zmodes(chamber)[1]

    wavelength!(laser1, pc.c/f_cl)
    detuning!(laser1, f_sb)
    polarization!(laser1, x̂)
    wavevector!(laser1, ẑ)

    wavelength!(laser2, pc.c/f_cl)
    detuning!(laser2, -f_sb)
    polarization!(laser2, x̂)
    wavevector!(laser2, ẑ)

    intensity!(laser1, A)
    intensity!(laser2, A)

    # setup the Hamiltonian
    h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf);
    # solve system
    tout = 0:0.1:t
    tout, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)
    
    SS = real(expect(ionprojector(chamber, "S", "S"), sol[end]))
    DD = real(expect(ionprojector(chamber, "D", "D"), sol[end]))
    SD = real(expect(ionprojector(chamber, "S", "D"), sol[end]))
    DS = real(expect(ionprojector(chamber, "D", "S"), sol[end]))

    correlator_values = Dict(1 => 1, 2 => -1, 3 => -1, 4 => 1)
    weights = [SS, SD, DS, DD]
    samples = sample(1:4, Weights(weights), N)
    values = [correlator_values[c] for c in samples]
    parity = 0
    
    for v in values
        if v == -1
            parity += 1
        end
    end

    P1 = parity/N
    delta_P1 = sqrt(P1*(1-P1)/N)

    ρ_red = ptrace(sol[end] ⊗ dagger(sol[end]), [3]);
    ρ = ionSim_compatibility(ρ_red)
    phase = 0:.1:π # Phase for the rotation. phase governs the axis of rotaation. for example, a phase of 0 is an x-rotation and a phase of π/2 is a y-rotation
    b = SpinBasis(1//2)
    SS = tensor(spinup(b), spinup(b)); SD = tensor(spinup(b), spindown(b)); 
    DS = tensor(spindown(b), spinup(b)); DD = tensor(spindown(b), spindown(b));
    measurements = zeros(length(phase))
    
    for (i, φ) in enumerate(phase)
        end_state = R(π/2, φ) * ρ * dagger(R(π/2, φ))
        probabilities = [proj(SS, end_state), proj(SD, end_state), proj(DS, end_state), proj(DD, end_state)]
        samples = sample(1:4, Weights(real(probabilities)), N)
        values = [correlator_values[c] for c in samples]
        parity = 0
        for v in values
            parity += v
        end
        parity /= N
        measurements[i] = parity[end]
    end

    model(phase, p) = p[1] * cos.(phase .* p[2] .+ p[3]) .+ p[4]

    # Perform the fit
    initial_params = [1.0, 1.0, 1.0, 0]  # Initial guesses for amplitude, frequency, phase offset, and baseline
    fit = curve_fit(model, phase, measurements, initial_params, lower = [-1, -2, -Inf, -.1], upper=[1.0, 2, Inf, .1])
    C = abs(fit.param[1])
    delta_C = sqrt(1/(N*length(phase)))

    fid = (1 - P1 + C) / 2
    delta_fid = sqrt((delta_P1^2 + delta_C^2)/4) # quantum projection noise and fit uncertainty

    return fid
end

# No Noise Version
# function Q(t, f_cl, f_sb, A, s=1000)
#     ca = Ca40([("S1/2", -1/2, "S"), ("D5/2", -1/2, "D")])
#     laser1 = Laser(pointing=[(1, 1.), (2, 1.)])
#     laser2 = Laser(pointing=[(1, 1.), (2, 1.)])
#     chain = LinearChain(
#             ions=[ca, ca], comfrequencies=(x=3e6,y=3e6,z=2.5e5), selectedmodes=(;z=[1])
#         )
#     chamber = Chamber(iontrap=chain, B=6e-4, Bhat=(x̂ + ẑ)/√2, lasers=[laser1, laser2]);
#     mode = zmodes(chamber)[1]

#     wavelength!(laser1, pc.c/f_cl)
#     detuning!(laser1, f_sb)
#     polarization!(laser1, x̂)
#     wavevector!(laser1, ẑ)

#     wavelength!(laser2, pc.c/f_cl)
#     detuning!(laser2, -f_sb)
#     polarization!(laser2, x̂)
#     wavevector!(laser2, ẑ)

#     intensity!(laser1, A)
#     intensity!(laser2, A)

#     h = hamiltonian(chamber, timescale=1e-6, lamb_dicke_order=1, rwa_cutoff=Inf)
#     tout = 0:0.01:t
#     _, sol = timeevolution.schroedinger_dynamic(tout, ca["S"] ⊗ ca["S"] ⊗ mode[0], h)
#     ρ = ptrace(sol[end] ⊗ dagger(sol[end]), [3]).data
#     return bell_fidelity_phi_plus(ρ)   # δφ = 0

# end

function trial(t, f_cl, f_sb, A; s=1000, tolerance=0.005, offset=2e3, step=1e2, max_iter=50)
    best_fid = Q(t, f_cl, f_sb, A)
    best_params = (f_cl, f_sb, A)
    iter = 0
    diff = Inf

    while diff > tolerance && iter < max_iter
        iter += 1
        prev_fid = best_fid

        # Search over f_cl
        f_cl_range = best_params[1] - offset:step:best_params[1] + offset
        for f_cl_trial in f_cl_range
            fid = Q(t, f_cl_trial, best_params[2], best_params[3])
            if fid > best_fid
                best_fid = fid
                best_params = (f_cl_trial, best_params[2], best_params[3])
            end
        end

        # Search over f_sb
        f_sb_range = best_params[2] - offset:step:best_params[2] + offset
        for f_sb_trial in f_sb_range
            fid = Q(t, best_params[1], f_sb_trial, best_params[3])
            if fid > best_fid
                best_fid = fid
                best_params = (best_params[1], f_sb_trial, best_params[3])
            end
        end

        # Search over A
        A_range = best_params[3] - offset:step:best_params[3] + offset
        for A_trial in A_range
            fid = Q(t, best_params[1], best_params[2], A_trial)
            if fid > best_fid
                best_fid = fid
                best_params = (best_params[1], best_params[2], A_trial)
            end
        end

        diff = abs(best_fid - prev_fid)
    end

    return [best_fid, t, best_params[1], best_params[2], best_params[3], iter]
end
