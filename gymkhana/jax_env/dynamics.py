"""JAX port of the STD drift dynamics used by PPO race training."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from gymkhana.envs.params import load_params


class JaxVehicleParams(NamedTuple):
    m: float
    lf: float
    lr: float
    h_s: float
    s_min: float
    s_max: float
    sv_min: float
    sv_max: float
    v_min: float
    v_max: float
    v_switch: float
    a_max: float
    I_z: float
    I_y_w: float
    R_w: float
    T_sb: float
    T_se: float
    T_steer: float
    tire_p_cx1: float
    tire_p_dx1: float
    tire_p_dx3: float
    tire_p_ex1: float
    tire_p_kx1: float
    tire_p_hx1: float
    tire_p_vx1: float
    tire_p_cy1: float
    tire_p_dy1: float
    tire_p_dy3: float
    tire_p_ey1: float
    tire_p_ky1: float
    tire_p_hy1: float
    tire_p_hy3: float
    tire_p_vy1: float
    tire_p_vy3: float
    tire_r_bx1: float
    tire_r_bx2: float
    tire_r_cx1: float
    tire_r_ex1: float
    tire_r_hx1: float
    tire_r_by1: float
    tire_r_by2: float
    tire_r_by3: float
    tire_r_cy1: float
    tire_r_ey1: float
    tire_r_hy1: float
    tire_r_vy1: float
    tire_r_vy3: float
    tire_r_vy4: float
    tire_r_vy5: float
    tire_r_vy6: float


def _params_from_dict(params: dict) -> JaxVehicleParams:
    fields = JaxVehicleParams._fields
    return JaxVehicleParams(**{field: float(params.get(field, 0.0)) for field in fields})


def load_vehicle_params(name: str = "f1tenth_std", overrides: dict | None = None) -> JaxVehicleParams:
    """Load vehicle params as a JAX pytree-friendly NamedTuple."""
    params = load_params(name)
    if overrides:
        params.update(overrides)
    return _params_from_dict(params)


def upper_accel_limit(vel, a_max, v_switch):
    return jnp.where(vel > v_switch, a_max * (v_switch / vel), a_max)


def accl_constraints(vel, a_long_d, v_switch, a_max, v_min, v_max):
    uac = upper_accel_limit(vel, a_max, v_switch)
    locked = ((vel <= v_min) & (a_long_d <= 0.0)) | ((vel >= v_max) & (a_long_d >= 0.0))
    return jnp.where(locked, 0.0, jnp.clip(a_long_d, -a_max, uac))


def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    locked = ((steering_angle <= s_min) & (steering_velocity <= 0.0)) | (
        (steering_angle >= s_max) & (steering_velocity >= 0.0)
    )
    return jnp.where(locked, 0.0, jnp.clip(steering_velocity, sv_min, sv_max))


def p_accl(speed, current_speed, max_a, max_v, min_v):
    vel_diff = speed - current_speed
    kp_forward_accel = 10.0 * max_a / max_v
    kp_forward_brake = 10.0 * max_a / (-min_v)
    kp_reverse_brake = 2.0 * max_a / max_v
    kp_reverse_accel = 2.0 * max_a / (-min_v)
    kp_forward = jnp.where(vel_diff > 0.0, kp_forward_accel, kp_forward_brake)
    kp_reverse = jnp.where(vel_diff > 0.0, kp_reverse_brake, kp_reverse_accel)
    return jnp.where(current_speed > 0.0, kp_forward, kp_reverse) * vel_diff


def speed_steering_angle_action(raw_action, x, params: JaxVehicleParams, dt: float):
    """Convert normalized ``[steer, speed]`` action into ``[sv, accel]``."""
    raw_steer = raw_action[..., 0]
    raw_speed = raw_action[..., 1]

    desired_angle = raw_steer * params.s_max
    k = jnp.where(
        params.T_steer > 0.0,
        (1.0 - jnp.exp(-dt / params.T_steer)) / dt,
        params.sv_max,
    )
    sv = k * (desired_angle - x[..., 2])

    v_center = 0.5 * (params.v_max + params.v_min)
    v_range = 0.5 * (params.v_max - params.v_min)
    desired_speed = raw_speed * v_range + v_center
    accel = p_accl(desired_speed, x[..., 3], params.a_max, params.v_max, params.v_min)

    return jnp.stack([sv, accel], axis=-1)


def formula_longitudinal(kappa, gamma, f_z, params: JaxVehicleParams):
    kappa = -kappa
    s_hx = params.tire_p_hx1
    s_vx = f_z * params.tire_p_vx1
    kappa_x = kappa + s_hx
    mu_x = params.tire_p_dx1 * (1.0 - params.tire_p_dx3 * gamma**2)
    c_x = params.tire_p_cx1
    d_x = mu_x * f_z
    e_x = params.tire_p_ex1
    k_x = f_z * params.tire_p_kx1
    b_x = k_x / (c_x * d_x)
    return d_x * jnp.sin(c_x * jnp.arctan(b_x * kappa_x - e_x * (b_x * kappa_x - jnp.arctan(b_x * kappa_x)))) + s_vx


def formula_lateral(alpha, gamma, f_z, params: JaxVehicleParams):
    s_hy = jnp.sign(gamma) * (params.tire_p_hy1 + params.tire_p_hy3 * jnp.abs(gamma))
    s_vy = jnp.sign(gamma) * f_z * (params.tire_p_vy1 + params.tire_p_vy3 * jnp.abs(gamma))
    alpha_y = alpha + s_hy
    mu_y = params.tire_p_dy1 * (1.0 - params.tire_p_dy3 * gamma**2)
    c_y = params.tire_p_cy1
    d_y = mu_y * f_z
    e_y = params.tire_p_ey1
    k_y = f_z * params.tire_p_ky1
    b_y = k_y / (c_y * d_y)
    f_y = d_y * jnp.sin(c_y * jnp.arctan(b_y * alpha_y - e_y * (b_y * alpha_y - jnp.arctan(b_y * alpha_y)))) + s_vy
    return f_y, mu_y


def formula_longitudinal_comb(kappa, alpha, f0_x, params: JaxVehicleParams):
    s_hxalpha = params.tire_r_hx1
    alpha_s = alpha + s_hxalpha
    b_xalpha = params.tire_r_bx1 * jnp.cos(jnp.arctan(params.tire_r_bx2 * kappa))
    c_xalpha = params.tire_r_cx1
    e_xalpha = params.tire_r_ex1
    d_xalpha = f0_x / (
        jnp.cos(
            c_xalpha
            * jnp.arctan(
                b_xalpha * s_hxalpha - e_xalpha * (b_xalpha * s_hxalpha - jnp.arctan(b_xalpha * s_hxalpha))
            )
        )
    )
    return d_xalpha * jnp.cos(
        c_xalpha * jnp.arctan(b_xalpha * alpha_s - e_xalpha * (b_xalpha * alpha_s - jnp.arctan(b_xalpha * alpha_s)))
    )


def formula_lateral_comb(kappa, alpha, gamma, mu_y, f_z, f0_y, params: JaxVehicleParams):
    s_hykappa = params.tire_r_hy1
    kappa_s = kappa + s_hykappa
    b_ykappa = params.tire_r_by1 * jnp.cos(jnp.arctan(params.tire_r_by2 * (alpha - params.tire_r_by3)))
    c_ykappa = params.tire_r_cy1
    e_ykappa = params.tire_r_ey1
    d_ykappa = f0_y / (
        jnp.cos(
            c_ykappa
            * jnp.arctan(
                b_ykappa * s_hykappa - e_ykappa * (b_ykappa * s_hykappa - jnp.arctan(b_ykappa * s_hykappa))
            )
        )
    )
    d_vykappa = mu_y * f_z * (params.tire_r_vy1 + params.tire_r_vy3 * gamma) * jnp.cos(
        jnp.arctan(params.tire_r_vy4 * alpha)
    )
    s_vykappa = d_vykappa * jnp.sin(params.tire_r_vy5 * jnp.arctan(params.tire_r_vy6 * kappa))
    return d_ykappa * jnp.cos(
        c_ykappa * jnp.arctan(b_ykappa * kappa_s - e_ykappa * (b_ykappa * kappa_s - jnp.arctan(b_ykappa * kappa_s)))
    ) + s_vykappa


def vehicle_dynamics_ks_cog(x, u_init, params: JaxVehicleParams):
    delta = x[..., 2]
    v = x[..., 3]
    psi = x[..., 4]
    lwb = params.lf + params.lr
    sv = steering_constraint(delta, u_init[..., 0], params.s_min, params.s_max, params.sv_min, params.sv_max)
    accel = accl_constraints(v, u_init[..., 1], params.v_switch, params.a_max, params.v_min, params.v_max)
    beta = jnp.arctan(jnp.tan(delta) * params.lr / lwb)
    return jnp.stack(
        [
            v * jnp.cos(beta + psi),
            v * jnp.sin(beta + psi),
            sv,
            accel,
            v * jnp.cos(beta) * jnp.tan(delta) / lwb,
        ],
        axis=-1,
    )


def vehicle_dynamics_std(x, u_init, params: JaxVehicleParams):
    x_pos = x[..., 0]
    y_pos = x[..., 1]
    delta = x[..., 2]
    v = x[..., 3]
    psi = x[..., 4]
    psi_dot = x[..., 5]
    beta = x[..., 6]
    omega_front = x[..., 7]
    omega_rear = x[..., 8]

    del x_pos, y_pos
    g = 9.81
    lwb = params.lf + params.lr
    v_s = 0.2
    v_b = 0.05
    v_min_blend = v_s / 2.0

    sv = steering_constraint(delta, u_init[..., 0], params.s_min, params.s_max, params.sv_min, params.sv_max)
    accel = accl_constraints(v, u_init[..., 1], params.v_switch, params.a_max, params.v_min, params.v_max)

    safe_v = jnp.maximum(v, v_min_blend)
    safe_v_cos_beta = jnp.maximum(v * jnp.cos(beta), v_min_blend)
    alpha_f_raw = jnp.arctan((v * jnp.sin(beta) + psi_dot * params.lf) / safe_v_cos_beta) - delta
    alpha_r_raw = jnp.arctan((v * jnp.sin(beta) - psi_dot * params.lr) / safe_v_cos_beta)
    alpha_f = jnp.where(v > v_min_blend, alpha_f_raw, 0.0)
    alpha_r = jnp.where(v > v_min_blend, alpha_r_raw, 0.0)

    f_zf = params.m * (-accel * params.h_s + g * params.lr) / (params.lr + params.lf)
    f_zr = params.m * (accel * params.h_s + g * params.lf) / (params.lr + params.lf)

    u_wf = jnp.maximum(
        0.0,
        v * jnp.cos(beta) * jnp.cos(delta) + (v * jnp.sin(beta) + params.lf * psi_dot) * jnp.sin(delta),
    )
    u_wr = jnp.maximum(0.0, v * jnp.cos(beta))

    s_f = 1.0 - params.R_w * omega_front / jnp.maximum(u_wf, v_min_blend)
    s_r = 1.0 - params.R_w * omega_rear / jnp.maximum(u_wr, v_min_blend)

    f0_xf = formula_longitudinal(s_f, 0.0, f_zf, params)
    f0_xr = formula_longitudinal(s_r, 0.0, f_zr, params)
    f0_yf, mu_yf = formula_lateral(alpha_f, 0.0, f_zf, params)
    f0_yr, mu_yr = formula_lateral(alpha_r, 0.0, f_zr, params)
    f_xf = formula_longitudinal_comb(s_f, alpha_f, f0_xf, params)
    f_xr = formula_longitudinal_comb(s_r, alpha_r, f0_xr, params)
    f_yf = formula_lateral_comb(s_f, alpha_f, 0.0, mu_yf, f_zf, f0_yf, params)
    f_yr = formula_lateral_comb(s_r, alpha_r, 0.0, mu_yr, f_zr, f0_yr, params)

    t_b = jnp.where(accel > 0.0, 0.0, params.m * params.R_w * accel)
    t_e = jnp.where(accel > 0.0, params.m * params.R_w * accel, 0.0)

    d_v = (
        -f_yf * jnp.sin(delta - beta)
        + f_yr * jnp.sin(beta)
        + f_xr * jnp.cos(beta)
        + f_xf * jnp.cos(delta - beta)
    ) / params.m
    dd_psi = (f_yf * jnp.cos(delta) * params.lf - f_yr * params.lr + f_xf * jnp.sin(delta) * params.lf) / params.I_z
    d_beta_raw = -psi_dot + (
        f_yf * jnp.cos(delta - beta)
        + f_yr * jnp.cos(beta)
        - f_xr * jnp.sin(beta)
        + f_xf * jnp.sin(delta - beta)
    ) / (params.m * safe_v)
    d_beta = jnp.where(v > v_min_blend, d_beta_raw, 0.0)

    d_omega_f = jnp.where(
        omega_front >= 0.0,
        (-params.R_w * f_xf + params.T_sb * t_b + params.T_se * t_e) / params.I_y_w,
        0.0,
    )
    d_omega_r = jnp.where(
        omega_rear >= 0.0,
        (-params.R_w * f_xr + (1.0 - params.T_sb) * t_b + (1.0 - params.T_se) * t_e) / params.I_y_w,
        0.0,
    )
    omega_front = jnp.maximum(0.0, omega_front)
    omega_rear = jnp.maximum(0.0, omega_rear)

    u_constrained = jnp.stack([sv, accel], axis=-1)
    x_ks = jnp.stack([x[..., 0], x[..., 1], delta, v, psi], axis=-1)
    f_ks = vehicle_dynamics_ks_cog(x_ks, u_constrained, params)
    d_beta_ks = (params.lr * sv) / (lwb * jnp.cos(delta) ** 2 * (1.0 + (jnp.tan(delta) ** 2 * params.lr / lwb) ** 2))
    dd_psi_ks = (
        accel * jnp.cos(beta) * jnp.tan(delta)
        - v * jnp.sin(beta) * d_beta_ks * jnp.tan(delta)
        + v * jnp.cos(beta) * sv / jnp.cos(delta) ** 2
    ) / lwb
    d_omega_f_ks = (u_wf / params.R_w - omega_front) / 0.02
    d_omega_r_ks = (u_wr / params.R_w - omega_rear) / 0.02

    w_std = 0.5 * (jnp.tanh((v - v_s) / v_b) + 1.0)
    w_ks = 1.0 - w_std
    return jnp.stack(
        [
            v * jnp.cos(beta + psi),
            v * jnp.sin(beta + psi),
            sv,
            w_std * d_v + w_ks * f_ks[..., 3],
            w_std * psi_dot + w_ks * f_ks[..., 4],
            w_std * dd_psi + w_ks * dd_psi_ks,
            w_std * d_beta + w_ks * d_beta_ks,
            w_std * d_omega_f + w_ks * d_omega_f_ks,
            w_std * d_omega_r + w_ks * d_omega_r_ks,
        ],
        axis=-1,
    )


def rk4_step(f, x, u, dt: float, params: JaxVehicleParams):
    k1 = f(x, u, params)
    k2 = f(x + 0.5 * dt * k1, u, params)
    k3 = f(x + 0.5 * dt * k2, u, params)
    k4 = f(x + dt * k3, u, params)
    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next.at[..., 4].set(jnp.mod(x_next[..., 4], 2.0 * jnp.pi))
