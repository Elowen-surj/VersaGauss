import warp as wp
from warp_utils import *
import numpy as np
import math


# compute stress from F
@wp.func
def kirchoff_stress_FCR(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    R = U * wp.transpose(V)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    return 2.0 * mu * (F - R) * wp.transpose(F) + id * lam * J * (J - 1.0)


@wp.func
def kirchoff_stress_neoHookean(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, J: float, sig: wp.vec3, mu: float, lam: float
):
    # compute kirchoff stress for FCR model (remember tau = P F^T)
    b = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    b_hat = b - wp.vec3(
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
        (b[0] + b[1] + b[2]) / 3.0,
    )
    tau = mu * J ** (-2.0 / 3.0) * b_hat + lam / 2.0 * (J * J - 1.0) * wp.vec3(
        1.0, 1.0, 1.0
    )
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )

@wp.func
def mls_mpm_snow(F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float, compression: float, stretch: float):
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    J = 1.0
    for d in range(0, 3):
        new_sig = wp.min(wp.max(sig[d], 1.0 - compression), 1.0 + stretch)
        J *= new_sig
    stress = 2.0 * mu * (F - U @ wp.transpose(V)) @ wp.transpose(F) + identity * lam * J * (J - 1.0)
    return stress

@wp.func
def mls_mpm_liquid(sig: wp.vec3, lam: float):
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    J = 1.0
    for d in range(0, 3):
        new_sig = sig[d]
        J *= new_sig
    stress = identity * lam * J * (J - 1.0)
    return stress

@wp.func
def mls_mpm_jelly(F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float):
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    J = 1.0
    for d in range(0, 3):
        new_sig = sig[d]
        J *= new_sig
    stress = 2.0 * mu * (F - U @ wp.transpose(V)) @ wp.transpose(F) + identity * lam * J * (J - 1.0)
    return stress

@wp.func
def kirchoff_stress_StVK(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    sig = wp.vec3(
        wp.max(sig[0], 0.01), wp.max(sig[1], 0.01), wp.max(sig[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    ONE = wp.vec3(1.0, 1.0, 1.0)
    tau = 2.0 * mu * epsilon + lam * log_sig_sum * ONE
    return (
        U
        * wp.mat33(tau[0], 0.0, 0.0, 0.0, tau[1], 0.0, 0.0, 0.0, tau[2])
        * wp.transpose(V)
        * wp.transpose(F)
    )

@wp.func 
def kirchoff_stress_water(
    J: float, bulk: float, gamma: float
):
    # gamma = 1.1 # gamma is set to be a liitle greater than 1 for weakly compressible fluids
    pressure = -bulk * (wp.pow(J, -gamma) - 1.)
    id = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    cauchy_stress = id * pressure
    return J * cauchy_stress


@wp.func
def kirchoff_stress_drucker_prager(
    F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float
):
    log_sig_sum = wp.log(sig[0]) + wp.log(sig[1]) + wp.log(sig[2])
    center00 = 2.0 * mu * wp.log(sig[0]) * (1.0 / sig[0]) + lam * log_sig_sum * (
        1.0 / sig[0]
    )
    center11 = 2.0 * mu * wp.log(sig[1]) * (1.0 / sig[1]) + lam * log_sig_sum * (
        1.0 / sig[1]
    )
    center22 = 2.0 * mu * wp.log(sig[2]) * (1.0 / sig[2]) + lam * log_sig_sum * (
        1.0 / sig[2]
    )
    center = wp.mat33(center00, 0.0, 0.0, 0.0, center11, 0.0, 0.0, 0.0, center22)
    return U * center * wp.transpose(V) * wp.transpose(F)

@wp.func
def sand_cal_stress(F: wp.mat33, U: wp.mat33, V: wp.mat33, sig: wp.vec3, mu: float, lam: float):
    sig_mat = wp.mat33(sig[0], 0.0, 0.0, 0.0, sig[1], 0.0, 0.0, 0.0, sig[2])
    inv_sig = wp.inverse(sig_mat)
    log_sig = wp.mat33(wp.log(wp.max(sig[0], 0.01)), 0.0, 0.0, 0.0, wp.log(wp.max(sig[1], 0.01)), 0.0, 0.0, 0.0, wp.log(wp.max(sig[2], 0.01)))
    log_sig_trace = log_sig[0,0] + log_sig[1,1] + log_sig[2,2]
    stress = U @ (2.0 * mu * inv_sig @ log_sig + lam * log_sig_trace * inv_sig) @ wp.transpose(V)
    stress = stress @ wp.transpose(F)
    # print(stress)
    return stress

@wp.func
def von_mises_return_mapping(F_trial: wp.mat33, model: MPMModelStruct, p: int, state: MPMStateStruct):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[state.particle_object_link[p]] * epsilon + model.lam[state.particle_object_link[p]] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[state.particle_object_link[p]]:
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[state.particle_object_link[p]] / (2.0 * model.mu[state.particle_object_link[p]])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        if model.hardening == 1:
            model.yield_stress[state.particle_object_link[p]] = (
                model.yield_stress[state.particle_object_link[p]] + 2.0 * model.mu[state.particle_object_link[p]] * model.xi[state.particle_object_link[p]] * delta_gamma
            )
        return F_elastic
    else:
        return F_trial


@wp.func
def von_mises_return_mapping_with_damage(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, state: MPMStateStruct
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    temp = (epsilon[0] + epsilon[1] + epsilon[2]) / 3.0

    tau = 2.0 * model.mu[state.particle_object_link[p]] * epsilon + model.lam[state.particle_object_link[p]] * (
        epsilon[0] + epsilon[1] + epsilon[2]
    ) * wp.vec3(1.0, 1.0, 1.0)
    sum_tau = tau[0] + tau[1] + tau[2]
    cond = wp.vec3(
        tau[0] - sum_tau / 3.0, tau[1] - sum_tau / 3.0, tau[2] - sum_tau / 3.0
    )
    if wp.length(cond) > model.yield_stress[state.particle_object_link[p]]:
        if model.yield_stress[state.particle_object_link[p]] <= 0:
            return F_trial
        epsilon_hat = epsilon - wp.vec3(temp, temp, temp)
        epsilon_hat_norm = wp.length(epsilon_hat) + 1e-6
        delta_gamma = epsilon_hat_norm - model.yield_stress[state.particle_object_link[p]] / (2.0 * model.mu[state.particle_object_link[p]])
        epsilon = epsilon - (delta_gamma / epsilon_hat_norm) * epsilon_hat
        model.yield_stress[state.particle_object_link[p]] = model.yield_stress[state.particle_object_link[p]] - 0.1 * wp.length(
            (delta_gamma / epsilon_hat_norm) * epsilon_hat
        )
        if model.yield_stress[state.particle_object_link[p]] <= 0:
            model.mu[state.particle_object_link[p]] = 0.0
            model.lam[state.particle_object_link[p]] = 0.0
        sig_elastic = wp.mat33(
            wp.exp(epsilon[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        # if model.hardening == 1:
        model.yield_stress[state.particle_object_link[p]] = (
            model.yield_stress[state.particle_object_link[p]] + 2.0 * model.mu[state.particle_object_link[p]] * model.xi[state.particle_object_link[p]] * delta_gamma
        )
        return F_elastic
    else:
        return F_trial


# for toothpaste
@wp.func
def viscoplasticity_return_mapping_with_StVK(
    F_trial: wp.mat33, model: MPMModelStruct, p: int, dt: float, state: MPMStateStruct
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_old = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig_old, V)

    sig = wp.vec3(
        wp.max(sig_old[0], 0.01), wp.max(sig_old[1], 0.01), wp.max(sig_old[2], 0.01)
    )  # add this to prevent NaN in extrem cases
    b_trial = wp.vec3(sig[0] * sig[0], sig[1] * sig[1], sig[2] * sig[2])
    epsilon = wp.vec3(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
    trace_epsilon = epsilon[0] + epsilon[1] + epsilon[2]
    epsilon_hat = epsilon - wp.vec3(
        trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
    )
    s_trial = 2.0 * model.mu[state.particle_object_link[p]] * epsilon_hat
    s_trial_norm = wp.length(s_trial)
    y = s_trial_norm - wp.sqrt(2.0 / 3.0) * model.yield_stress[state.particle_object_link[p]]
    if y > 0:
        mu_hat = model.mu[state.particle_object_link[p]] * (b_trial[0] + b_trial[1] + b_trial[2]) / 3.0
        s_new_norm = s_trial_norm - y / (
            1.0 + 0.0 / (2.0 * mu_hat * dt)    # plastic_viscosity
        )
        s_new = (s_new_norm / s_trial_norm) * s_trial
        epsilon_new = 1.0 / (2.0 * model.mu[state.particle_object_link[p]]) * s_new + wp.vec3(
            trace_epsilon / 3.0, trace_epsilon / 3.0, trace_epsilon / 3.0
        )
        sig_elastic = wp.mat33(
            wp.exp(epsilon_new[0]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[1]),
            0.0,
            0.0,
            0.0,
            wp.exp(epsilon_new[2]),
        )
        F_elastic = U * sig_elastic * wp.transpose(V)
        return F_elastic
    else:
        return F_trial

@wp.func
def mls_mpm_snow_f(F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int, dt: float):
    # base = wp.int(state.particle_x[p] * model.inv_dx - 0.5)
    # fx = state.particle_x[p] * model.inv_dx - wp.float(base)
    # w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # F_trial = (identity + dt * state.particle_C[p]) @ F_trial
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_mat = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)
    
    J = 1.0
    for d in range(0, 3):
        new_sig = wp.min(wp.max(sig[d], 1.0 - model.compression), 1.0 + model.stretch)
        state.particle_Jp[p] = state.particle_Jp[p] * sig[d] / new_sig
        sig[d] = new_sig
        J *= new_sig
        sig_mat[d, d] = sig[d]
    F_trial = U @ sig_mat @ wp.transpose(V)

    return F_trial

@wp.func
def mls_mpm_liquid_f(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int, dt: float
):
    identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    F_trial = (identity + dt * state.particle_C[p]) @ F_trial
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig_mat = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)
    
    J = 1.0
    for d in range(0, 3):
        new_sig = wp.min(wp.max(sig[d], 1.0 - 2.5e-2), 1.0 + 4.5e-3)
        J *= new_sig
    F_trial = identity * wp.sqrt(J)
    
    return F_trial
    

@wp.func
def sand_return_mapping(
    F_trial: wp.mat33, state: MPMStateStruct, model: MPMModelStruct, p: int
):
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    wp.svd3(F_trial, U, sig, V)

    epsilon = wp.vec3(
        wp.log(wp.max(wp.abs(sig[0]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[1]), 1e-14)),
        wp.log(wp.max(wp.abs(sig[2]), 1e-14)),
    )
    # print(model.sand_s[2,p])
    # epsilon = wp.mat33(
    #     wp.log(wp.max(sig[0], 0.01)), 0.0, 0.0, 0.0, wp.log(wp.max(sig[1], 0.01)), 0.0, 0.0, 0.0, wp.log(wp.max(sig[2], 0.01))
    # )

    # identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    # eps = epsilon + model.sand_s[1, p] / 3.0 * identity
    # eps = eps + (-0.01 * (1.0 - model.sand_s[0, p])) / (3.0 * model.sand_s[2, p]) * identity
    # eps_trace = eps[0,0] + eps[1,1] + eps[2,2]
    # ehat = eps - eps_trace / 3.0 * identity
    # Fnorm = wp.sqrt(wp.max(wp.pow(ehat[0,0], 2.0) + wp.pow(ehat[1,1], 2.0) + wp.pow(ehat[2,2], 2.0), 0.01))
    # yp = Fnorm + (3.0 * model.lam[state.particle_object_link[p]] + 2.0 * model.mu[state.particle_object_link[p]]) / (2.0 * model.mu[state.particle_object_link[p]]) * eps_trace * model.sand_s[2, p]
    # new_e = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # delta_q = 0.0
    # if Fnorm <= 1e-10 or eps_trace > 1e-10:
    #     delta_q = wp.sqrt(wp.pow(eps[0, 0], 2.0) + wp.pow(eps[1, 1], 2.0) + wp.pow(eps[2, 2], 2.0))
    # elif yp <= 1e-10:
    #     new_e = epsilon
    #     delta_q = 0.0
    # else:
    #     new_e = eps - yp / Fnorm * ehat
    #     delta_q = yp

    # model.sand_s[3, p] = model.sand_s[3, p] + delta_q
    # phi = 35.0 + (9.0 * model.sand_s[3, p] - 10.0) * wp.exp(-0.2 * model.sand_s[3, p])
    # phi = phi / 180.0 * 3.14159265358979
    # sin_phi = wp.sin(phi)
    # model.sand_s[2, p] = wp.sqrt(2.0 / 3.0) * (2.0 * sin_phi) / (3.0 - sin_phi)
    # new_F = U @ wp.mat33(wp.exp(new_e[0, 0]), 0.0, 0.0, 0.0, wp.exp(new_e[1, 1]), 0.0, 0.0, 0.0, wp.exp(new_e[2, 2]))

    # model.sand_s[1, p] = model.sand_s[1, p] - wp.log(wp.max(0.01, wp.determinant(new_F))) + wp.log(wp.max(0.01, wp.determinant(state.particle_F[p])))
    # # # print(new_F)
    # return new_F

    sigma_out = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    tr = epsilon[0] + epsilon[1] + epsilon[2]  # + state.particle_Jp[p]
    epsilon_hat = epsilon - wp.vec3(tr / 3.0, tr / 3.0, tr / 3.0)
    epsilon_hat_norm = wp.length(epsilon_hat)
    delta_gamma = (
        epsilon_hat_norm
        + (3.0 * model.lam[state.particle_object_link[p]] + 2.0 * model.mu[state.particle_object_link[p]])
        / (2.0 * model.mu[state.particle_object_link[p]])
        * tr
        * model.alpha
    )

    if delta_gamma <= 0:
        F_elastic = F_trial

    if delta_gamma > 0 and tr > 0:
        F_elastic = U * wp.transpose(V)

    if delta_gamma > 0 and tr <= 0:
        H = epsilon - epsilon_hat * (delta_gamma / epsilon_hat_norm)
        s_new = wp.vec3(wp.exp(H[0]), wp.exp(H[1]), wp.exp(H[2]))

        F_elastic = U * wp.diag(s_new) * wp.transpose(V)
    return F_elastic


@wp.kernel
def compute_mu_lam_from_E_nu(state: MPMStateStruct, model: MPMModelStruct):
    p = wp.tid()
    model.mu[p] = model.E[p] / (2.0 * (1.0 + model.nu[p]))
    model.lam[p] = (
        model.E[p] * model.nu[p] / ((1.0 + model.nu[p]) * (1.0 - 2.0 * model.nu[p]))
    )


@wp.kernel
def zero_grid(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    for i in range(0, model.object_nums+1):
        state.grid_m[i, grid_x, grid_y, grid_z] = 0.0
        state.grid_vol[i, grid_x, grid_y, grid_z] = 0.0
        state.grid_data[i * 2, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
        state.grid_data[i * 2 + 1, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
        state.grid_v[i, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
        state.grid_v_in[i*2, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
        state.grid_v_in[i*2+1, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_shs[0, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)
    state.grid_shs[1, grid_x, grid_y, grid_z] = wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_dweight(
    model: MPMModelStruct, w: wp.mat33, dw: wp.mat33, i: int, j: int, k: int
):
    dweight = wp.vec3(
        dw[0, i] * w[1, j] * w[2, k],
        w[0, i] * dw[1, j] * w[2, k],
        w[0, i] * w[1, j] * dw[2, k],
    )
    return dweight * model.inv_dx


@wp.func
def update_cov(state: MPMStateStruct, p: int, grad_v: wp.mat33, dt: float):
    cov_n = wp.mat33(0.0)
    cov_n[0, 0] = state.particle_init_cov[p * 6]
    cov_n[0, 1] = state.particle_init_cov[p * 6 + 1]
    cov_n[0, 2] = state.particle_init_cov[p * 6 + 2]
    cov_n[1, 0] = state.particle_init_cov[p * 6 + 1]
    cov_n[1, 1] = state.particle_init_cov[p * 6 + 3]
    cov_n[1, 2] = state.particle_init_cov[p * 6 + 4]
    cov_n[2, 0] = state.particle_init_cov[p * 6 + 2]
    cov_n[2, 1] = state.particle_init_cov[p * 6 + 4]
    cov_n[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov_np1 = cov_n + dt * (grad_v * cov_n + cov_n * wp.transpose(grad_v))

    state.particle_init_cov[p * 6] = cov_np1[0, 0]
    state.particle_init_cov[p * 6 + 1] = cov_np1[0, 1]
    state.particle_init_cov[p * 6 + 2] = cov_np1[0, 2]
    state.particle_init_cov[p * 6 + 3] = cov_np1[1, 1]
    state.particle_init_cov[p * 6 + 4] = cov_np1[1, 2]
    state.particle_init_cov[p * 6 + 5] = cov_np1[2, 2]


@wp.kernel
def p2g_apic_with_stress(state: MPMStateStruct, model: MPMModelStruct, dt: float):
    # input given to p2g:   particle_stress
    #                       particle_x
    #                       particle_v
    #                       particle_C
    p = wp.tid()
    # if state.particle_selection[p] == 0:
    stress = state.particle_stress[p]
    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
    identity = wp.vec3(1.0, 0.0, 0.0)

    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                dpos = (
                    wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                ) * model.dx
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                # print(weight)
                dweight = compute_dweight(model, w, dw, i, j, k)
                C = state.particle_C[p]
                # if model.rpic = 0, standard apic
                C = (1.0 - model.rpic_damping[state.particle_object_link[p]]) * C + model.rpic_damping[state.particle_object_link[p]] / 2.0 * (
                    C - wp.transpose(C)
                )
                if model.rpic_damping[state.particle_object_link[p]] < -0.001:
                    # standard pic
                    C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

                elastic_force = -state.particle_vol[p] * stress * dweight

                # if model.material[p] == 4 or model.material[p] == 6 or model.material[p] == 0:
                # if model.material[state.particle_object_link[p]] == 4 or model.material[state.particle_object_link[p]] == 6 or model.material[state.particle_object_link[p]] == 2:
                if model.material[state.particle_object_link[p]] == 4 or model.material[state.particle_object_link[p]] == 6: 
                # if model.material[p] == 4:
                    #     affine = stress + state.particle_mass[p] * state.particle_C[p]
                    # else:
                    #     affine = stress + state.particle_mass[p] * state.particle_C[p]
                    v_in_add_p = (
                        weight
                        * state.particle_mass[p] * (state.particle_v[p] + state.particle_C[p] @ dpos)
                    ) # P^n_(im)
                    v_in_add_f = (
                        weight * stress @ dpos
                    ) # f^n_(im)
                    # v_in_add = (
                    #     weight
                    #     * (state.particle_mass[p]
                    #     * state.particle_v[p]
                    #     + affine @ dpos)
                    # )
                    # if model.material[state.particle_object_link[p]] == 2:
                    #     print(v_in_add_f)
                else:
                    
                    # v_in_add = (
                    #     weight
                    #     * state.particle_mass[p]
                    #     * (state.particle_v[p] + C * dpos)
                    #     + dt * elastic_force
                    # )
                    v_in_add_p = (
                        weight * state.particle_mass[p] * (state.particle_v[p] + C * dpos)
                    ) # P^n_(im)
                    v_in_add_f = dt * elastic_force # f^n_(im)
                
                # wp.atomic_add(state.grid_v_in, model.object_nums * 2, ix, iy, iz, v_in_add_p + v_in_add_f)

                # wp.atomic_add(
                #     state.grid_m, model.object_nums, ix, iy, iz, weight * state.particle_mass[p]
                # )    # m_(im)^n

                # wp.atomic_add(state.grid_v_in, model.object_nums * 2, ix, iy, iz, v_in_add_p)
                
                # wp.atomic_add(state.grid_v_in, model.object_nums * 2 + 1, ix, iy, iz, v_in_add_f)

                wp.atomic_add(state.grid_v_in, state.particle_object_link[p] * 2, ix, iy, iz, v_in_add_p)

                wp.atomic_add(state.grid_v_in, state.particle_object_link[p] * 2 + 1, ix, iy, iz, v_in_add_f)
                
                # wp.atomic_add(
                #     state.grid_m, model.object_nums, ix, iy, iz, weight * state.particle_mass[p]
                # )    # m_(im)^n
                wp.atomic_add(
                    state.grid_m, state.particle_object_link[p], ix, iy, iz, weight * state.particle_mass[p]
                )   # m_(ik)^n
                
                wp.atomic_add(
                    state.grid_vol, state.particle_object_link[p], ix, iy, iz, weight * state.particle_vol[p]
                )  # V_(ik)^n
                
                # wp.atomic_add(
                #     state.grid_vol, model.object_nums, ix, iy, iz, weight * state.particle_vol[p]
                # ) # V_(im)^n

                if model.material[state.particle_object_link[p]] == 6:
                    wp.atomic_add(
                        state.grid_shs, 0, ix, iy, iz, weight * state.particle_init_shs[p]
                    )
                    wp.atomic_add(
                        state.grid_shs, 1, ix, iy, iz, weight * identity
                    )
                
                wp.atomic_add(
                    state.grid_data[state.particle_object_link[p] * 2 + 1], ix, iy, iz, dweight * state.particle_mass[p]
                )
                
                wp.atomic_add(
                    state.grid_data[state.particle_object_link[p] * 2], ix, iy, iz, dweight * state.particle_vol[p]
                )  # dV_(ik)^n
                
                # wp.atomic_add(
                #     state.grid_dvol, state.particle_object_link[p], ix, iy, iz, dweight * state.particle_vol[p]
                # )  # dV_(ik)^n

@wp.kernel
def grid_objects_sum(state: MPMStateStruct, model: MPMModelStruct):
    grid_x, grid_y, grid_z = wp.tid()
    for i in range(0, model.object_nums):
        state.grid_v_in[model.object_nums * 2, grid_x, grid_y, grid_z] += state.grid_v_in[i * 2, grid_x, grid_y, grid_z]
        state.grid_v_in[model.object_nums * 2 + 1, grid_x, grid_y, grid_z] += state.grid_v_in[i * 2 + 1, grid_x, grid_y, grid_z]
        
        state.grid_m[model.object_nums, grid_x, grid_y, grid_z] += state.grid_m[i, grid_x, grid_y, grid_z]
        state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] += state.grid_vol[i, grid_x, grid_y, grid_z]
        state.grid_data[model.object_nums * 2, grid_x, grid_y, grid_z] += state.grid_data[i * 2, grid_x, grid_y, grid_z]

# add gravity
@wp.kernel
def grid_normalization_and_gravity(
    state: MPMStateStruct, model: MPMModelStruct, dt: float, eta: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[model.object_nums, grid_x, grid_y, grid_z] > eta:
        # print(grid_x)
        v_out = state.grid_v_in[model.object_nums * 2, grid_x, grid_y, grid_z] * (
            1.0 / state.grid_m[model.object_nums, grid_x, grid_y, grid_z]
        )
        # state.grid_v_out[model.object_nums, grid_x, grid_y, grid_z] = v_out
        # v_out = v_out + dt * model.gravitational_accelaration
        # state.grid_v[0, grid_x, grid_y, grid_z] = v_out


        v_in_1 = state.grid_v_in[model.object_nums * 2 + 1, grid_x, grid_y, grid_z]
        if state.grid_shs[1, grid_x, grid_y, grid_z][0] > 1e-6:
            state.grid_shs[0, grid_x, grid_y, grid_z] = state.grid_shs[0, grid_x, grid_y, grid_z] * (1.0 / state.grid_shs[1, grid_x, grid_y, grid_z][0])
        for i in range(0, model.object_nums):
            if state.grid_m[i, grid_x, grid_y, grid_z] > eta:
            # if state.grid_m[model.object_nums, grid_x, grid_y, grid_z] > eta:
                # if i == 1:
                #     print(state.grid_v_in[i * 2 + 1, grid_x, grid_y, grid_z])
                    # print(state.grid_v_in[3, grid_x, grid_y, grid_z])
                # v_out = state.grid_v_in[i * 2, grid_x, grid_y, grid_z] * (
                #     1.0 / state.grid_m[i, grid_x, grid_y, grid_z]
                # )
                # v_out = v_out + dt * model.gravitational_accelaration + v_in_1 * (1.0 / state.grid_m[i, grid_x, grid_y, grid_z])
                    # print(state.grid_v_in[i * 2 + 1, grid_x, grid_y, grid_z])
                    # print((1.0 / state.grid_m[i, grid_x, grid_y, grid_z]))
                    # print(state.grid_v_in[i * 2 + 1, grid_x, grid_y, grid_z] * (1.0 / state.grid_m[i, grid_x, grid_y, grid_z]))
                
                a = state.grid_vol[i, grid_x, grid_y, grid_z] * (1.0 / state.grid_vol[model.object_nums, grid_x, grid_y, grid_z])
                v_out += (v_in_1 * a) * (1.0 / state.grid_m[i, grid_x, grid_y, grid_z]) + dt * model.gravitational_accelaration
                
                # v_out += state.grid_v_in[i * 2 + 1, grid_x, grid_y, grid_z] * (1.0 / state.grid_m[i, grid_x, grid_y, grid_z]) + dt * model.gravitational_accelaration

                # v_out += v_in_1 * (1.0 / state.grid_m[model.object_nums, grid_x, grid_y, grid_z]) + dt * model.gravitational_accelaration

                # add gravity
                # else:
                # v_out = v_out + v_in_1 * (1.0 / state.grid_m[model.object_nums, grid_x, grid_y, grid_z])
                # v_out = v_out + dt * model.gravitational_accelaration
                state.grid_v[i, grid_x, grid_y, grid_z] = v_out

@wp.kernel
def sand_water_mixture(
    state: MPMStateStruct, model: MPMModelStruct, dt: float
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] > 1e-15:
        # print('aaa')
        for k in range(0, model.object_nums):
            if model.material[k] == 6 and state.grid_vol[k, grid_x, grid_y, grid_z] > 1e-15:     # material == fluid
                # print('bbb')
                for k1 in range(0, model.object_nums):
                    if k1 != k and model.material[k1] == 2 and state.grid_vol[k1, grid_x, grid_y, grid_z] > 1e-15:    #material == sand
                        # print('ccc')
                        water_density = state.grid_m[k, grid_x, grid_y, grid_z] / state.grid_vol[k, grid_x, grid_y, grid_z]
                        # sand_density = 400.0
                        cE = (wp.pow(model.porosity, 3.0) * water_density * model.gravitational_accelaration[2]) / model.permeability
                        M = wp.mat22(state.grid_m[k1, grid_x, grid_y, grid_z], 0.0, 0.0, state.grid_m[k, grid_x, grid_y, grid_z])
                        d = cE * state.grid_m[k1, grid_x, grid_y, grid_z] * state.grid_m[k, grid_x, grid_y, grid_z]
                        D = wp.mat22(-d, d, d, -d)
                        A = M + dt * D
                        A_inv = wp.inverse(A)
                        v_fluid = state.grid_v_in[k * 2, grid_x, grid_y, grid_z] * (1.0 / state.grid_m[k, grid_x, grid_y, grid_z])
                        v_sand = state.grid_v_in[k1 * 2, grid_x, grid_y, grid_z] * (1.0 / state.grid_m[k1, grid_x, grid_y, grid_z])
                        B_1 = M[0, 0] * v_sand + dt * M[0, 0] * model.gravitational_accelaration + state.grid_v_in[k1 * 2 + 1, grid_x, grid_y, grid_z]
                        B_2 = M[1, 1] * v_fluid + dt * M[1, 1] * model.gravitational_accelaration + state.grid_v_in[k * 2 + 1, grid_x, grid_y, grid_z]

                        # grid_v_ori = state.grid_v[k1, grid_x, grid_y, grid_z]
                        state.grid_v[k1, grid_x, grid_y, grid_z] = A_inv[0,0] * B_1 + A_inv[0,1] * B_2
                        # print(state.grid_v[k1, grid_x, grid_y, grid_z] - grid_v_ori)

                        state.grid_v[k, grid_x, grid_y, grid_z] = A_inv[1,0] * B_1 + A_inv[1,1] * B_2


# @wp.kernel
# def grid_diffusion(
#     state: MPMStateStruct, model: MPMModelStruct, judge: wp.array(dtype=int), theta: float
# ):
#     grid_x, grid_y, grid_z = wp.tid()
#     if state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] > 1e-15:
#         for j in range(0, model.object_nums):
#             if model.material[j] == 6 and state.grid_vol[j, grid_x, grid_y, grid_z] > 1e-15:
#                 for t in range(0, model.object_nums):
#                     if model.material[t] == 6 and state.grid_vol[t, grid_x, grid_y, grid_z] > 1e-15 and t != j:
#                         judge[j] = 1
            
#         for k in range(0, model.object_nums):
#             if judge[k] == 1 and state.grid_m[model.object_nums, grid_x, grid_y, grid_z] > 1e-15:
#                 a_k = state.grid_vol[k, grid_x, grid_y, grid_z] * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z]
#                 da_k = state.grid_data[k * 2, grid_x, grid_y, grid_z] * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] - state.grid_vol[k, grid_x, grid_y, grid_z] * state.grid_data[model.object_nums * 2, grid_x, grid_y, grid_z]
#                 v_diff_1 = da_k * (1.0 / a_k)
#                 v_diff_2 = wp.vec3(0.0, 0.0, 0.0)
#                 for k1 in range(0, model.object_nums):
#                     if k1 != k and model.material[k1] == 6 and state.grid_vol[k1, grid_x, grid_y, grid_z] > 1e-15:
#                         c_k1 = state.grid_m[k1, grid_x, grid_y, grid_z] * (1.0 / state.grid_m[model.object_nums, grid_x, grid_y, grid_z])
#                         a_k1 = state.grid_vol[k1, grid_x, grid_y, grid_z] * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z]
#                         da_k1 = state.grid_data[k1 * 2, grid_x, grid_y, grid_z]  * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] - state.grid_vol[k1, grid_x, grid_y, grid_z] * state.grid_data[model.object_nums * 2, grid_x, grid_y, grid_z]
#                         v_diff_2 += c_k1 * da_k1 * (1.0 / a_k1)
#                 v_diff = model.diff_coe[k] * (v_diff_1 - v_diff_2)
#                 v_diff[2] = v_diff[2] * theta
#                 state.grid_v[k, grid_x, grid_y, grid_z] = state.grid_v[k, grid_x, grid_y, grid_z] + v_diff

@wp.kernel
def grid_diffusion(
    state: MPMStateStruct, model: MPMModelStruct, judge: wp.array(dtype=int)
):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] > 1e-15:
        for j in range(0, model.object_nums):
            if model.material[j] == 6 and state.grid_vol[j, grid_x, grid_y, grid_z] > 1e-15:
                for t in range(0, model.object_nums):
                    if model.material[t] == 6 and state.grid_vol[t, grid_x, grid_y, grid_z] > 1e-15 and t != j:
                        judge[j] = 1
            
        for k in range(0, model.object_nums):
            if judge[k] == 1 and state.grid_m[model.object_nums, grid_x, grid_y, grid_z] > 1e-15:
                a_k = state.grid_vol[k, grid_x, grid_y, grid_z] * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z]
                da_k = state.grid_data[k * 2, grid_x, grid_y, grid_z] * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] - state.grid_vol[k, grid_x, grid_y, grid_z] * state.grid_data[model.object_nums * 2, grid_x, grid_y, grid_z]
                v_diff_1 = da_k * (1.0 / a_k)
                v_diff_2 = wp.vec3(0.0, 0.0, 0.0)
                for k1 in range(0, model.object_nums):
                    if k1 != k and model.material[k1] == 6 and state.grid_vol[k1, grid_x, grid_y, grid_z] > 1e-15:
                        c_k1 = state.grid_m[k1, grid_x, grid_y, grid_z] * (1.0 / state.grid_m[model.object_nums, grid_x, grid_y, grid_z])
                        a_k1 = state.grid_vol[k1, grid_x, grid_y, grid_z] * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z]
                        da_k1 = state.grid_data[k1 * 2, grid_x, grid_y, grid_z]  * state.grid_vol[model.object_nums, grid_x, grid_y, grid_z] - state.grid_vol[k1, grid_x, grid_y, grid_z] * state.grid_data[model.object_nums * 2, grid_x, grid_y, grid_z]
                        v_diff_2 += c_k1 * da_k1 * (1.0 / a_k1)
                v_diff = model.diff_coe[k] * (v_diff_1 - v_diff_2)
                state.grid_v[k, grid_x, grid_y, grid_z] = state.grid_v[k, grid_x, grid_y, grid_z] + v_diff

@wp.kernel
def solid_fluid_inter(state: MPMStateStruct, model: MPMModelStruct, beta: float):
    grid_x, grid_y, grid_z = wp.tid()
    if state.grid_m[model.object_nums, grid_x, grid_y, grid_z] > 1e-15:
        # print(grid_x)
        for k in range(0, model.object_nums):
            if model.material[k] == 6 and state.grid_m[k, grid_x, grid_y, grid_z] > 1e-15:
                # wp.printf("bbbbbbbbbbbbbbbbbbbbbb")
                for k1 in range(0, model.object_nums):
                    if model.material[k1] == 0 and state.grid_vol[k1, grid_x, grid_y, grid_z] > 1e-15:
                        # wp.printf("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                        grid_mf = state.grid_data[k * 2 + 1, grid_x, grid_y, grid_z]
                        grid_ms = state.grid_data[k1 * 2 + 1, grid_x, grid_y, grid_z]
                        grid_mf_l = wp.sqrt(grid_mf[0] ** 2.0 + grid_mf[1] ** 2.0 + grid_mf[2] ** 2.0)
                        grid_mf_s = wp.sqrt(grid_ms[0] ** 2.0 + grid_ms[1] ** 2.0 + grid_ms[2] ** 2.0)
                        if grid_mf_l > 1e-15 and grid_mf_s > 1e-15:
                            nf = grid_mf * (1.0 / grid_mf_l)
                            ns = grid_ms * (1.0 / grid_mf_s)
                            # beta = 1.0
                            fco = (state.grid_v_in[k * 2, grid_x, grid_y, grid_z] + state.grid_v_in[k * 2 + 1, grid_x, grid_y, grid_z]) * state.grid_m[k, grid_x, grid_y, grid_z] - (state.grid_v_in[k1 * 2, grid_x, grid_y, grid_z] + state.grid_v_in[k1 * 2 + 1, grid_x, grid_y, grid_z]) * state.grid_m[k1, grid_x, grid_y, grid_z]
                            
                            fco = fco * (1.0 / (state.grid_m[k, grid_x, grid_y, grid_z] + state.grid_m[k1, grid_x, grid_y, grid_z]))
                            
                            n_diff = nf - ns
                            n_diff_l = wp.sqrt(n_diff[0] ** 2.0 + n_diff[1] ** 2.0 + n_diff[2] ** 2.0)
                            if n_diff_l > 1e-15:
                                nI = n_diff * (1.0 / n_diff_l)
                                # wp.printf("nI: %f\n", nI)
                                fco_nI = (fco[0] * nI[0] + fco[1] * nI[1] + fco[2] * nI[2]) * nI
                                state.grid_v[k, grid_x, grid_y, grid_z] = state.grid_v[k, grid_x, grid_y, grid_z] + beta * fco_nI * (1.0 / state.grid_m[model.object_nums, grid_x, grid_y, grid_z])
                                
                                state.grid_v[k1, grid_x, grid_y, grid_z] = state.grid_v[k1, grid_x, grid_y, grid_z] - beta * fco_nI * (1.0 / state.grid_m[model.object_nums, grid_x, grid_y, grid_z])

# @wp.kernel
# def compute_energy(state: MPMStateStruct, model: MPMModelStruct):
#     # mv^2, phi*V, mgx
#     p = wp.tid()
#     # model.energy = model.energy + 100.0
#     state.energy[p] = 0.0
#     for i in range(3):
#         state.energy[p] = state.energy[p] + (state.particle_mass[p] * state.particle_v[p][i] * state.particle_v[p][i])
#         # if state.particle_mass[p] * state.particle_v[p][i] * state.particle_v[p][i] < 0.0:
#         #     print(model.energy)
#     state.energy[p] = state.energy[p] + (state.particle_mass[p] * (- model.gravitational_accelaration[2]) * (state.particle_x[p][2] - 0.55))
#     # if state.particle_mass[p] * (- model.gravitational_accelaration[2]) * (state.particle_x[p][2] - 0.5) < 0.0:
#     #     print(model.energy)
#     # print(model.energy)
#     p_link = state.particle_object_link[p]
#     energy = 0.0
#     if model.material[p_link] == 0:
#         U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#         V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
#         sig = wp.vec3(0.0)
#         wp.svd3(state.particle_F[p], U, sig, V)
#         ss = 0.0
#         J = wp.determinant(state.particle_F[p])
#         for i in range(3):
#             ss += (sig[i] - 1.0) * (sig[i] - 1.0)
#         energy = model.mu[p_link] * ss + model.lam[p_link] * 0.5 * (J - 1.0) * (J - 1.0) 
#     if model.material[p_link] == 6:
#         energy = model.Ef[p_link] * ((wp.pow(state.particle_Jf[p], 1.0 - model.gamma) / (model.gamma - 1.0)) + state.particle_Jf[p])
#     # if energy < 0.0:
#     #     print(energy)
#     state.energy[p] = state.energy[p] + energy * state.particle_vol[p]


@wp.kernel
def g2p(state: MPMStateStruct, model: MPMModelStruct, dt: float, n_particles: int):
    p = wp.tid()
    # if state.particle_selection[p] == 0:
    grid_pos = state.particle_x[p] * model.inv_dx
    base_pos_x = wp.int(grid_pos[0] - 0.5)
    base_pos_y = wp.int(grid_pos[1] - 0.5)
    base_pos_z = wp.int(grid_pos[2] - 0.5)
    fx = grid_pos - wp.vec3(
        wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
    )
    wa = wp.vec3(1.5) - fx
    wb = fx - wp.vec3(1.0)
    wc = fx - wp.vec3(0.5)
    w = wp.mat33(
        wp.cw_mul(wa, wa) * 0.5,
        wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
        wp.cw_mul(wc, wc) * 0.5,
    )
    dw = wp.mat33(fx - wp.vec3(1.5), -2.0 * (fx - wp.vec3(1.0)), fx - wp.vec3(0.5))
    new_v = wp.vec3(0.0, 0.0, 0.0)
    new_C = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    new_F = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    # new_shs = wp.vec3(0.0, 0.0, 0.0)
    # if model.material[state.particle_object_link[p]] == 2:
    #     model.sand_s[0, p] = 0.0
    # particle_weight = 0.0
    for i in range(0, 3):
        for j in range(0, 3):
            for k in range(0, 3):
                ix = base_pos_x + i
                iy = base_pos_y + j
                iz = base_pos_z + k
                dpos = wp.vec3(wp.float(i), wp.float(j), wp.float(k)) - fx
                weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                grid_v = state.grid_v[state.particle_object_link[p], ix, iy, iz]

                # grid_v = state.grid_v[0, ix, iy, iz]


                new_v = new_v + grid_v * weight
                new_C = new_C + wp.outer(grid_v, dpos) * (
                    weight * model.inv_dx * 4.0
                )
                dweight = compute_dweight(model, w, dw, i, j, k)
                new_F = new_F + wp.outer(grid_v, dweight)
                # if model.material[state.particle_object_link[p]] == 2: # sand
                #     for k1 in range(model.object_nums):
                #         if model.material[k1] == 6: # fluid
                #             if state.grid_m[state.particle_object_link[p], ix, iy, iz] > 1e-15 and state.grid_m[k1, ix, iy, iz] > 1e-15:
                #                 model.sand_s[0, p] = model.sand_s[0, p] + weight
                # if model.material[state.particle_object_link[p]] == 6:
                #     new_shs = new_shs + state.grid_shs[0, ix, iy, iz] * weight
                #     particle_weight = particle_weight + weight

    I33 = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    state.particle_v[p] = new_v
    # if state.particle_object_link[p] == 1:
    state.particle_x[p] = state.particle_x[p] + dt * new_v
    state.particle_C[p] = new_C
    # if model.material[state.particle_object_link[p]] == 6:
    #     state.particle_init_shs[n_particles + p] = new_shs * (1.0 / particle_weight)
    # if model.material[state.particle_object_link[p]] == 4 or model.material[state.particle_object_link[p]] == 6 or model.material[state.particle_object_link[p]] == 2:
    if model.material[state.particle_object_link[p]] == 4 or model.material[state.particle_object_link[p]] == 6: 
        F_tmp = (I33 + state.particle_C[p] * dt) @ state.particle_F[p]
    else:
        F_tmp = (I33 + new_F * dt) @ state.particle_F[p]
    # if model.material[state.particle_object_link[p]] != 6:
    state.particle_F[p] = F_tmp
    new_C_trace = 0.0
    for d in range(0, 3):
        new_C_trace += new_C[d, d]
    state.particle_Jf[p] = state.particle_Jf[p] * (1.0 + dt * new_C_trace)
    # state.particle_Jp[p] *= 1 + dt * wp.trace(new_C)


    if model.update_cov_with_F:
        if model.material[state.particle_object_link[p]] == 6:
            identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
            deformation_grad = state.particle_Jf[p] * identity
        else:
            deformation_grad = state.particle_F[p]
        update_cov(state, p, deformation_grad, dt)

    # state.energy[p] = state.energy[p] + 100.0


# compute (Kirchhoff) stress = stress(returnMap(F_trial))
@wp.kernel
def compute_stress_from_F_trial(
    state: MPMStateStruct, model: MPMModelStruct, dt: float, fluid_type: bool
):
    p = wp.tid()
    p_link = state.particle_object_link[p]
    # print(state.particle_object_link[p])
    # if state.particle_selection[p] == 0:
    # apply return mapping
    if model.material[p_link] == 1:  # metal
        state.particle_F[p] = von_mises_return_mapping(
            state.particle_F[p], model, p, state
        )
    if model.material[p_link] == 2:  # sand
        state.particle_F[p] = sand_return_mapping(
            state.particle_F[p], state, model, p
        )
    elif model.material[p_link] == 3:  # visplas, with StVk+VM, no thickening foam
        state.particle_F[p] = viscoplasticity_return_mapping_with_StVK(
            state.particle_F[p], model, p, dt, state
        )
    elif model.material[p_link] == 5:
        state.particle_F[p] = von_mises_return_mapping_with_damage(
            state.particle_F[p], model, p, state
        )
    elif model.material[p_link] == 4:
        state.particle_F[p] = mls_mpm_snow_f(state.particle_F[p], state, model, p, dt)
    # elif model.material[p] == 6:
    #     state.particle_F[p] = mls_mpm_liquid_f(state.particle_F[p], state, model, p, dt)
    else:  # elastic
        state.particle_F[p] = state.particle_F[p]

    # also compute stress here
    J = wp.determinant(state.particle_F[p])
    U = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    V = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    sig = wp.vec3(0.0)
    stress = wp.mat33(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    wp.svd3(state.particle_F[p], U, sig, V)
    if model.material[p_link] == 5 or model.material[p_link] == 0:
        stress = kirchoff_stress_FCR(
            state.particle_F[p], U, V, J, model.mu[state.particle_object_link[p]], model.lam[state.particle_object_link[p]]
        )
    if model.material[p_link] == 1:
        stress = kirchoff_stress_StVK(
            state.particle_F[p], U, V, sig, model.mu[state.particle_object_link[p]], model.lam[state.particle_object_link[p]]
        )
    if model.material[p_link] == 2:
        stress = kirchoff_stress_drucker_prager(
            state.particle_F[p], U, V, sig, model.mu[state.particle_object_link[p]], model.lam[state.particle_object_link[p]]
        )
        # stress = sand_cal_stress(state.particle_F[p], U, V, sig, model.mu[state.particle_object_link[p]], model.lam[state.particle_object_link[p]])
        # print(stress)
    if model.material[p_link] == 3:
        # temporarily use stvk, subject to change
        stress = kirchoff_stress_StVK(
            state.particle_F[p], U, V, sig, model.mu[state.particle_object_link[p]], model.lam[state.particle_object_link[p]]
        )
    if model.material[p_link] == 4:
        h = wp.exp(model.hardening_coe * (1.0 - state.particle_Jp[p]))
        mu, lam = model.mu[state.particle_object_link[p]] * h, model.lam[state.particle_object_link[p]] * h
        stress = mls_mpm_snow(state.particle_F[p], U, V, sig, mu, lam, model.compression, model.stretch)
    
    # if model.material[p] == 0:
    #     h = 0.3
    #     mu, lam = model.mu[state.particle_object_link[p]] * h, model.lam[state.particle_object_link[p]] * h
    #     stress = mls_mpm_jelly(state.particle_F[p], U, V, sig, mu, lam)
    
    if model.material[p_link] == 6:
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        # stress = (-dt * 4.0 * model.Ef[p] * state.particle_vol[p] * (state.particle_Jf[p] - 1.0) * model.inv_dx * model.inv_dx) * identity * 10.0
        if fluid_type:
            # stress = model.Ef[state.particle_object_link[p]] * (state.particle_Jf[state.particle_object_link[p]] - 1.0) * identity
            h = wp.exp(10.0 * (1.0 - state.particle_Jp[p]))
            la = model.lam[state.particle_object_link[p]] * h
            stress = mls_mpm_liquid(sig, la)
        
        # h = wp.exp(10.0 * (1.0 - state.particle_Jp[p]))
        # la = model.lam[state.particle_object_link[p]] * h
        # gamma = 8.0
        # print(model.gamma)
        else:
            stress = model.Ef[state.particle_object_link[p]] * (1.0 - 1.0 / (state.particle_Jf[p] ** model.gamma)) * state.particle_Jf[p] * identity

        # stress = kirchoff_stress_water(
        #         J, model.Ef[p], model.gamma
        #     )

    if model.material[p_link] == 7:
        stress = kirchoff_stress_water(
                J, model.Ef[state.particle_object_link[p]], model.gamma
            )

    # if model.material[p] == 4 or model.material[p] == 0:
    # if model.material[p] == 4:
    # if model.material[p_link] == 4 or model.material[p_link] == 6 or model.material[p_link] == 2:
    if model.material[p_link] == 4 or model.material[p_link] == 6: 
        stress = (-dt * 4.0 * state.particle_vol[p] * model.inv_dx * model.inv_dx) * stress
    else:
        stress = (stress + wp.transpose(stress)) / 2.0  # enfore symmetry
    state.particle_stress[p] = stress


@wp.kernel
def compute_cov_from_F(state: MPMStateStruct, model: MPMModelStruct, particle_cov: wp.array(dtype=float)):
    p = wp.tid()

    if model.material[state.particle_object_link[p]] == 6 or model.material[state.particle_object_link[p]] == 7:
        identity = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        # def_p = wp.min(wp.max(wp.pow(state.particle_Jf[p], 1.0/3.0), -3.0), 3.0)
        F = state.particle_Jf[p] * identity
        # F = def_p * identity
        
        # F = identity
    else:
        F = state.particle_F[p]

    init_cov = wp.mat33(0.0)
    init_cov[0, 0] = state.particle_init_cov[p * 6]
    init_cov[0, 1] = state.particle_init_cov[p * 6 + 1]
    init_cov[0, 2] = state.particle_init_cov[p * 6 + 2]
    init_cov[1, 0] = state.particle_init_cov[p * 6 + 1]
    init_cov[1, 1] = state.particle_init_cov[p * 6 + 3]
    init_cov[1, 2] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 0] = state.particle_init_cov[p * 6 + 2]
    init_cov[2, 1] = state.particle_init_cov[p * 6 + 4]
    init_cov[2, 2] = state.particle_init_cov[p * 6 + 5]

    cov = F * init_cov * wp.transpose(F)
    # cov = init_cov


    # state.particle_cov[p * 6] = wp.min(cov[0, 0], wp.pow(wp.pow(state.particle_vol[p], 1.0 / 3.0) / 4.0, 0.5))
    # state.particle_cov[p * 6 + 1] = wp.min(cov[0, 1], wp.pow(wp.pow(state.particle_vol[p], 1.0 / 3.0) / 4.0, 0.5))
    # state.particle_cov[p * 6 + 2] = wp.min(cov[0, 2], wp.pow(wp.pow(state.particle_vol[p], 1.0 / 3.0) / 4.0, 0.5))
    # state.particle_cov[p * 6 + 3] = wp.min(cov[1, 1], wp.pow(wp.pow(state.particle_vol[p], 1.0 / 3.0) / 4.0, 0.5))
    # state.particle_cov[p * 6 + 4] = wp.min(cov[1, 2], wp.pow(wp.pow(state.particle_vol[p], 1.0 / 3.0) / 4.0, 0.5))
    # state.particle_cov[p * 6 + 5] = wp.min(cov[2, 2], wp.pow(wp.pow(state.particle_vol[p], 1.0 / 3.0) / 4.0, 0.5))
    # cov_scale_mean = (cov[0, 0] + cov[1, 1] + cov[2, 2]) /3 

    # state.particle_cov[p * 6] = cov[0, 0]
    # state.particle_cov[p * 6 + 1] = cov[0, 1]
    # state.particle_cov[p * 6 + 2] = cov[0, 2]
    # state.particle_cov[p * 6 + 3] = cov[1, 1]
    # state.particle_cov[p * 6 + 4] = cov[1, 2]
    # state.particle_cov[p * 6 + 5] = cov[2, 2]
    particle_cov[p * 6] = cov[0, 0]
    particle_cov[p * 6 + 1] = cov[0, 1]
    particle_cov[p * 6 + 2] = cov[0, 2]
    particle_cov[p * 6 + 3] = cov[1, 1]
    particle_cov[p * 6 + 4] = cov[1, 2]
    particle_cov[p * 6 + 5] = cov[2, 2]
    # particle_cov[p * 6] = state.particle_init_cov[p * 6]
    # particle_cov[p * 6 + 1] = state.particle_init_cov[p * 6 + 1]
    # particle_cov[p * 6 + 2] = state.particle_init_cov[p * 6 + 2]
    # particle_cov[p * 6 + 3] = state.particle_init_cov[p * 6 + 3]
    # particle_cov[p * 6 + 4] = state.particle_init_cov[p * 6 + 4]
    # particle_cov[p * 6 + 5] = state.particle_init_cov[p * 6 + 5]

@wp.kernel
def compute_shs_from_grid(state: MPMStateStruct, model: MPMModelStruct, particle_shs: wp.array(dtype=wp.vec3)):
    p = wp.tid()

    if model.material[state.particle_object_link[p]] == 6:
        grid_pos = state.particle_x[p] * model.inv_dx
        base_pos_x = wp.int(grid_pos[0] - 0.5)
        base_pos_y = wp.int(grid_pos[1] - 0.5)
        base_pos_z = wp.int(grid_pos[2] - 0.5)
        fx = grid_pos - wp.vec3(
            wp.float(base_pos_x), wp.float(base_pos_y), wp.float(base_pos_z)
        )
        wa = wp.vec3(1.5) - fx
        wb = fx - wp.vec3(1.0)
        wc = fx - wp.vec3(0.5)
        w = wp.mat33(
            wp.cw_mul(wa, wa) * 0.5,
            wp.vec3(0.0, 0.0, 0.0) - wp.cw_mul(wb, wb) + wp.vec3(0.75),
            wp.cw_mul(wc, wc) * 0.5,
        )
        new_shs = wp.vec3(0.0, 0.0, 0.0)
        particle_weight = 0.0
        for i in range(0, 3):
            for j in range(0, 3):
                for k in range(0, 3):
                    ix = base_pos_x + i
                    iy = base_pos_y + j
                    iz = base_pos_z + k
                    weight = w[0, i] * w[1, j] * w[2, k]  # tricubic interpolation
                    new_shs = new_shs + state.grid_shs[0, ix, iy, iz] * weight
                    particle_weight = particle_weight + weight
        particle_shs[p] = new_shs * (1.0 / particle_weight)
    else:
        particle_shs[p] = state.particle_init_shs[p]


# @wp.kernel
# def compute_R_from_F(state: MPMStateStruct, model: MPMModelStruct):
#     p = wp.tid()

#     F = state.particle_F[p]

#     # polar svd decomposition
#     U = wp.mat33(0.0)
#     V = wp.mat33(0.0)
#     sig = wp.vec3(0.0)
#     wp.svd3(F, U, sig, V)

#     if wp.determinant(U) < 0.0:
#         U[0, 2] = -U[0, 2]
#         U[1, 2] = -U[1, 2]
#         U[2, 2] = -U[2, 2]

#     if wp.determinant(V) < 0.0:
#         V[0, 2] = -V[0, 2]
#         V[1, 2] = -V[1, 2]
#         V[2, 2] = -V[2, 2]

#     # compute rotation matrix
#     R = U * wp.transpose(V)
#     state.particle_R[p] = wp.transpose(R)


@wp.kernel
def add_damping_via_grid(state: MPMStateStruct, scale: float):
    k, grid_x, grid_y, grid_z = wp.tid()
    state.grid_v[k, grid_x, grid_y, grid_z] = (
        state.grid_v[k, grid_x, grid_y, grid_z] * scale
    )


@wp.kernel
def apply_additional_params(
    state: MPMStateStruct,
    model: MPMModelStruct,
    params_modifier: MaterialParamsModifier,
):
    p = wp.tid()
    pos = state.particle_x[p]
    if (
        pos[0] > params_modifier.point[0] - params_modifier.size[0]
        and pos[0] < params_modifier.point[0] + params_modifier.size[0]
        and pos[1] > params_modifier.point[1] - params_modifier.size[1]
        and pos[1] < params_modifier.point[1] + params_modifier.size[1]
        and pos[2] > params_modifier.point[2] - params_modifier.size[2]
        and pos[2] < params_modifier.point[2] + params_modifier.size[2]
    ):
        model.E[state.particle_object_link[p]] = params_modifier.E
        model.nu[state.particle_object_link[p]] = params_modifier.nu
        state.particle_density[p] = params_modifier.density


@wp.kernel
def selection_add_impulse_on_particles(
    state: MPMStateStruct, impulse_modifier: Impulse_modifier
):
    p = wp.tid()
    offset = state.particle_x[p] - impulse_modifier.point
    if (
        wp.abs(offset[0]) < impulse_modifier.size[0]
        and wp.abs(offset[1]) < impulse_modifier.size[1]
        and wp.abs(offset[2]) < impulse_modifier.size[2]
    ):
        impulse_modifier.mask[p] = 1
    else:
        impulse_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
        and state.particle_object_link[p] == velocity_modifier.object_index
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0
        
@wp.kernel
def selection_enforce_object_particle_velocity_translation(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point
    if (
        wp.abs(offset[0]) < velocity_modifier.size[0]
        and wp.abs(offset[1]) < velocity_modifier.size[1]
        and wp.abs(offset[2]) < velocity_modifier.size[2]
        and state.particle_object_link[p] == velocity_modifier.object_index
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0


@wp.kernel
def selection_enforce_particle_velocity_cylinder(
    state: MPMStateStruct, velocity_modifier: ParticleVelocityModifier
):
    p = wp.tid()
    offset = state.particle_x[p] - velocity_modifier.point

    vertical_distance = wp.abs(wp.dot(offset, velocity_modifier.normal))

    horizontal_distance = wp.length(
        offset - wp.dot(offset, velocity_modifier.normal) * velocity_modifier.normal
    )
    if (
        state.particle_object_link[p] == velocity_modifier.object_index and
        vertical_distance < velocity_modifier.half_height_and_radius[0]
        and horizontal_distance < velocity_modifier.half_height_and_radius[1]
    ):
        velocity_modifier.mask[p] = 1
    else:
        velocity_modifier.mask[p] = 0
