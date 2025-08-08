
import numpy as np
from .physics import (
    mph_to_mps, mps_to_mph, apparent_wind_speed, wing_cl_ramp,
    eta_vs_speed, wing_forces, water_drag, foil_lift_max
)

def can_fly_for_wind(true_wind_mps, params):
    v = np.linspace(0.1, params['v_max_mps'], 240)
    v_app = apparent_wind_speed(true_wind_mps, v, params['rel_angle_deg'])
    cl_eff = wing_cl_ramp(v_app, params['wing_cl_max'], params['vaw_roll_in'])
    pump = np.where(v < 3.0, params['skill']['pump_mult'], 1.0)
    eta_curve = eta_vs_speed(v, params['eta_low_bias'],
                             params['wing_eta_high'] * params['skill']['eta_mult'],
                             params['eta_half_v'])
    F_thrust, F_vert = wing_forces(v_app, params['wing_area_m2'], cl_eff * pump,
                                   eta_curve, params['vertical_fraction'], params['rho_air'])
    D = water_drag(v, params['rho_water'], params['CdA_taxi'], params['low_speed_taxi'],
                   params['v_hump'], params['sigma'], skill_drag_mult=params['skill']['drag_mult'])
    W_eff_arr = np.maximum(params['weight_N'] - F_vert, 0.0)
    L_foil = foil_lift_max(v, params['rho_water'], params['foil_area_m2'],
                           params['foil_cl_max'], params['foil_eff_mult_total'])
    vertical_ok = L_foil >= (W_eff_arr * params['lift_safety'])
    horizontal_ok = F_thrust >= (D * params['taxis_margin'])
    both = vertical_ok & horizontal_ok

    if both.any():
        idx = np.argmax(both)
        v_takeoff = v[idx]
        feasible_indices = np.where(both)[0]
        diffs = np.abs(F_thrust[feasible_indices] - D[feasible_indices])
        eq_idx = feasible_indices[np.argmin(diffs)]
        v_eq = v[eq_idx]
        return True, float(v_takeoff), float(v_eq)
    return False, None, None

def min_wind_threshold_mph(params, w_min_mph=1.0, w_max_mph=45.0, step_mph=0.25):
    winds_mph = np.arange(w_min_mph, w_max_mph + step_mph, step_mph)
    for W_mph in winds_mph:
        ok, _, _ = can_fly_for_wind(mph_to_mps(W_mph), params)
        if ok:
            return float(W_mph)
    return None
