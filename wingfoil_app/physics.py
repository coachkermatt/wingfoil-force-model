
import numpy as np

def mph_to_mps(mph: float) -> float:
    return mph * 0.44704

def mps_to_mph(mps: float) -> float:
    return mps / 0.44704

def apparent_wind_speed(true_wind_mps, board_speed_mps, rel_angle_deg=90.0):
    theta = np.deg2rad(rel_angle_deg)
    true = np.asarray(true_wind_mps, dtype=float)
    board = np.asarray(board_speed_mps, dtype=float)
    return np.sqrt(true*true + board*board - 2.0*true*board*np.cos(theta))

def wing_cl_ramp(v_app, cl_max, vaw_roll_in):
    if vaw_roll_in <= 0:
        return np.full_like(v_app, cl_max, dtype=float) if hasattr(v_app, '__len__') else cl_max
    return cl_max * (1.0 - np.exp(-np.asarray(v_app) / vaw_roll_in))

def eta_vs_speed(v, eta_low_bias, eta_high, eta_half_v):
    v = np.asarray(v)
    eta_low = eta_high * eta_low_bias
    return eta_low + (eta_high - eta_low) * (v*v) / (v*v + eta_half_v*eta_half_v + 1e-9)

def wing_forces(v_app, area_m2, cl_eff, eta, vertical_fraction, rho_air):
    F = 0.5 * rho_air * area_m2 * cl_eff * (np.asarray(v_app)**2)
    F_vert = F * vertical_fraction
    F_thrust = F * (1.0 - vertical_fraction) * eta
    return F_thrust, F_vert

def water_drag(v, rho_water, CdA, hump_mag, v_hump, sigma, skill_drag_mult=1.0):
    v = np.asarray(v)
    base = 0.5 * rho_water * CdA * v*v
    hump = hump_mag * np.exp(-((v - v_hump)**2) / (2.0 * sigma*sigma))
    return (base * (1.0 + hump)) * skill_drag_mult

def foil_lift_max(v, rho_water, area_m2, cl_max_foil, eff_mult=1.0):
    v = np.asarray(v)
    return 0.5 * rho_water * area_m2 * cl_max_foil * eff_mult * (v*v)
