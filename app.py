# Streamlit app (MPH-only UI)
import json, math, os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

st.set_page_config(page_title="Wing Foiling Force Model — AXIS + Ozone (MPH)", layout="wide")
load_dotenv()

# -------------------------
# Data / Gear
# -------------------------
GEAR_PATH = Path(__file__).parent / "gear.json"
with open(GEAR_PATH, "r") as f:
    GEAR = json.load(f)

# Skill presets
SKILL = {
    "Beginner":     {"eta_mult": 0.85, "pump_mult": 0.90, "drag_mult": 1.15},
    "Intermediate": {"eta_mult": 0.95, "pump_mult": 1.00, "drag_mult": 1.05},
    "Advanced":     {"eta_mult": 1.05, "pump_mult": 1.05, "drag_mult": 1.00},
    "Expert":       {"eta_mult": 1.10, "pump_mult": 1.10, "drag_mult": 0.95},
}

# -------------------------
# Unit helpers
# -------------------------
def mph_to_mps(mph): return mph * 0.44704
def mps_to_mph(mps): return mps / 0.44704

# -------------------------
# Physics helpers
# -------------------------

def apparent_wind_speed(true_wind_mps, board_speed_mps, rel_angle_deg=90.0):
    # Vectorized: works for scalars or numpy arrays
    theta = np.deg2rad(rel_angle_deg)
    true = np.asarray(true_wind_mps, dtype=float)
    board = np.asarray(board_speed_mps, dtype=float)
    return np.sqrt(true*true + board*board - 2.0*true*board*np.cos(theta))

def wing_cl_ramp(v_app, cl_max, vaw_roll_in):
    # Smooth ramp towards cl_max: CL_eff = cl_max * (1 - exp(-Vapp / vaw_roll_in))
    if vaw_roll_in <= 0:
        return np.full_like(v_app, cl_max, dtype=float) if hasattr(v_app, "__len__") else cl_max
    return cl_max * (1.0 - np.exp(-np.asarray(v_app) / vaw_roll_in))

def eta_vs_speed(v, eta_low_bias, eta_high, eta_half_v):
    # Logistic-ish ramp: eta = eta_low + (eta_high - eta_low) * v^2 / (v^2 + v_half^2)
    v = np.asarray(v)
    eta_low = eta_high * eta_low_bias
    return eta_low + (eta_high - eta_low) * (v*v) / (v*v + eta_half_v*eta_half_v + 1e-9)

def wing_forces(v_app, area_m2, cl_eff, eta, vertical_fraction, rho_air):
    # Total aerodynamic force magnitude ~ 0.5*rho_air*A*CL*V^2
    F = 0.5 * rho_air * area_m2 * cl_eff * (np.asarray(v_app)**2)
    # Split into vertical vs horizontal, then apply efficiency to horizontal "useful thrust"
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

def can_fly_for_wind(true_wind_mps, params):
    # Evaluate both gates across speeds; return (bool, v_takeoff_mps, v_equil_mps)
    v = np.linspace(0.1, params["v_max_mps"], 240)
    v_app = apparent_wind_speed(true_wind_mps, v, params["rel_angle_deg"])
    cl_eff = wing_cl_ramp(v_app, params["wing_cl_max"], params["vaw_roll_in"])

    # Skill pumping boost at sub-3 m/s
    pump = np.where(v < 3.0, params["skill"]["pump_mult"], 1.0)

    eta_curve = eta_vs_speed(v, params["eta_low_bias"],
                             params["wing_eta_high"] * params["skill"]["eta_mult"],
                             params["eta_half_v"])
    F_thrust, F_vert = wing_forces(v_app, params["wing_area_m2"], cl_eff * pump,
                                   eta_curve, params["vertical_fraction"], params["rho_air"])

    D = water_drag(v, params["rho_water"], params["CdA_taxi"], params["low_speed_taxi"],
                   params["v_hump"], params["sigma"], skill_drag_mult=params["skill"]["drag_mult"])

    W_eff_arr = np.maximum(params["weight_N"] - F_vert, 0.0)
    L_foil = foil_lift_max(v, params["rho_water"], params["foil_area_m2"],
                           params["foil_cl_max"], params["foil_eff_mult_total"])

    vertical_ok = L_foil >= (W_eff_arr * params["lift_safety"])
    horizontal_ok = F_thrust >= (D * params["taxis_margin"])
    both = vertical_ok & horizontal_ok

    if both.any():
        idx = np.argmax(both)  # first True
        v_takeoff = v[idx]
        # equilibrium: where thrust ~ drag inside feasible region; pick closest diff
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

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Inputs")

# Rider / Conditions (MPH only)
wind_mph = st.sidebar.number_input("True wind (mph)", min_value=0.0, value=18.0, step=0.5)
wind_mps = mph_to_mps(wind_mph)

rel_angle_deg = st.sidebar.slider("Relative wind angle (deg)", 30, 150, 90, help="90° ~ beam reach; lower values skew upwind.")

weight_lbs = st.sidebar.number_input("Rider weight (lb)", min_value=60.0, value=165.0, step=1.0)
weight_kg = weight_lbs * 0.453592
weight_N = weight_kg * 9.81

skill_name = st.sidebar.selectbox("Skill", list(SKILL.keys()), index=2)
skill = SKILL[skill_name]

# Gear selection
st.sidebar.subheader("Wing (handheld)")
brand = "Ozone"
model = st.sidebar.selectbox("Model", list(GEAR["wings"][brand].keys()), index=0)
sizes = GEAR["wings"][brand][model]["sizes_m2"]
wing_size = st.sidebar.select_slider("Size (m²)", options=sizes, value=5.0)
wing_cl_max = GEAR["wings"][brand][model]["cl_max"]
wing_eta_high = GEAR["wings"][brand][model]["eta_high"]
wing_area_m2 = wing_size

st.sidebar.subheader("Foil (AXIS)")
fronts = GEAR["foils"]["AXIS"]["front_wings"]
front_labels = [f'{fw["model"]} ({fw["area_cm2"]} cm²)' for fw in fronts]
front_idx = st.sidebar.selectbox("Front wing", list(range(len(fronts))), format_func=lambda i: front_labels[i], index=1)
front = fronts[front_idx]

stabs = GEAR["foils"]["AXIS"]["stabilizers"]
stab_labels = [f'{s["model"]} ({s["area_cm2"]} cm²)' for s in stabs]
stab_idx = st.sidebar.selectbox("Stabilizer", list(range(len(stabs))), format_func=lambda i: stab_labels[i], index=1)
stab = stabs[stab_idx]

fuses = GEAR["foils"]["AXIS"]["fuselages"]
fuse_labels = [f'{f["model"]}' for f in fuses]
fuse_idx = st.sidebar.selectbox("Fuselage", list(range(len(fuses))), format_func=lambda i: fuse_labels[i], index=1)
fuse = fuses[fuse_idx]

# Display options
st.sidebar.subheader("Display & Scan")
# Accept v_max in mph for the UI; convert internally
v_max_mph = st.sidebar.slider("Max board speed to scan (mph)", 10.0, 40.0, 27.0, 1.0)
v_max_mps = mph_to_mps(v_max_mph)
mobile_ui = st.sidebar.checkbox("Mobile UI (larger touch targets)", value=False)

# Physics + safety dials
with st.sidebar.expander("Physics & Safety (advanced)", expanded=False):
    rho_air = st.number_input("Air density (kg/m³)", value=1.225, step=0.01, min_value=0.8, max_value=1.35)
    rho_water = st.number_input("Water density (kg/m³)", value=1025.0, step=1.0, min_value=990.0, max_value=1050.0)
    vaw_roll_in = st.number_input("Wing CL roll-in (m/s)", value=3.0, step=0.1, help="Bigger = slower CL ramp.")
    eta_low_bias = st.number_input("η low-speed bias (0–1)", value=0.5, step=0.05, min_value=0.0, max_value=1.0)
    eta_half_v = st.number_input("η half-speed (m/s)", value=3.0, step=0.1)
    vertical_fraction = st.number_input("Wing vertical fraction (0–0.6)", value=0.20, step=0.01, min_value=0.0, max_value=0.6)
    foil_cl_max = st.number_input("Foil CL_max", value=0.60, step=0.02, min_value=0.3, max_value=1.0)
    CdA_taxi = st.number_input("CdA_taxi (m²)", value=0.08, step=0.005, min_value=0.01, max_value=0.2)
    low_speed_taxi = st.number_input("Low-speed taxi hump", value=1.5, step=0.1, min_value=0.0, max_value=3.0)
    v_hump = st.number_input("Hump peak speed (m/s)", value=1.5, step=0.1, min_value=0.5, max_value=4.0)
    sigma = st.number_input("Hump width σ (m/s)", value=0.8, step=0.1, min_value=0.1, max_value=3.0)
    lift_safety = st.number_input("Lift safety margin", value=1.05, step=0.01, min_value=1.0, max_value=1.3)
    taxis_margin = st.number_input("Taxi thrust margin", value=1.10, step=0.01, min_value=1.0, max_value=1.5)

# Session/params
if mobile_ui:
    st.markdown("""
        <style>
        .stSlider > div { padding-top: 0.5rem; padding-bottom: 0.5rem; }
        .stButton > button { padding: 0.8rem 1.2rem; font-size: 1.05rem; }
        </style>
    """, unsafe_allow_html=True)

params = dict(
    rel_angle_deg = float(rel_angle_deg),
    v_max_mps = float(v_max_mps),
    weight_N = float(weight_N),
    skill = skill,
    wing_area_m2 = float(wing_area_m2),
    wing_cl_max = float(wing_cl_max),
    wing_eta_high = float(wing_eta_high),
    vaw_roll_in = float(vaw_roll_in),
    eta_low_bias = float(eta_low_bias),
    eta_half_v = float(eta_half_v),
    vertical_fraction = float(vertical_fraction),
    foil_area_m2 = float(front["area_cm2"] / 1e4),
    foil_cl_max = float(foil_cl_max),
    foil_eff_mult_total = float(front.get("eff_mult", 1.0) * stab.get("eff_mult", 1.0) * fuse.get("eff_mult", 1.0)),
    CdA_taxi = float(CdA_taxi),
    low_speed_taxi = float(low_speed_taxi),
    v_hump = float(v_hump),
    sigma = float(sigma),
    lift_safety = float(lift_safety),
    taxis_margin = float(taxis_margin),
    rho_air = float(rho_air),
    rho_water = float(rho_water),
)

# -------------------------
# Smart Suggestions
# -------------------------
def evaluate_can_fly(true_wind_mps, params):
    ok, v_to_mps, v_eq_mps = can_fly_for_wind(true_wind_mps, params)
    return ok, v_to_mps, v_eq_mps

def suggest_adjustment(true_wind_mps, params):
    ok, v_to_mps, v_eq_mps = evaluate_can_fly(true_wind_mps, params)
    current_size = wing_size

    def test_size(new_size):
        p = params.copy()
        p["wing_area_m2"] = float(new_size)
        return evaluate_can_fly(true_wind_mps, p)[0]

    available = sorted(sizes)
    if not ok:
        # Find smallest upsize that works
        for s in available:
            if s > current_size and test_size(s):
                return f"Upsize wing to **{s:.1f} m²** to fly.", {"wing_size": s}
        # If no wing size fixes it, try next-larger front wing by area
        fronts_sorted = sorted(fronts, key=lambda fw: fw["area_cm2"])
        cur_area = front["area_cm2"]
        for fw in fronts_sorted:
            if fw["area_cm2"] > cur_area:
                p = params.copy()
                p["foil_area_m2"] = fw["area_cm2"] / 1e4
                p["foil_eff_mult_total"] = fw.get("eff_mult", 1.0) * stab.get("eff_mult", 1.0) * fuse.get("eff_mult", 1.0)
                if evaluate_can_fly(true_wind_mps, p)[0]:
                    return f"Swap to **{fw['model']} ({fw['area_cm2']} cm²)** front wing.", {"front_wing": fw["model"]}
        return "Consider **both** a larger wing **and** a larger front wing.", {}
    else:
        # Tasteful downsizing if possible
        downs = [s for s in available if s < current_size]
        for s in reversed(downs):
            if test_size(s):
                return f"You can downsize to **{s:.1f} m²** and still fly.", {"wing_size": s}
        return "Your current gear is well-matched. Optional: tune safety margins or η for feel.", {}

# -------------------------
# Layout & UI
# -------------------------
st.title("Wing Foiling Force Model — AXIS + Ozone (MPH)")

# KPI cards
ok, v_takeoff_mps, v_eq_mps = evaluate_can_fly(wind_mps, params)
wind_floor_mph = min_wind_threshold_mph(params)

col1, col2, col3, col4 = st.columns(4)
col1.metric("True wind", f"{wind_mph:.1f} mph")
col2.metric("Takeoff possible", "Yes" if ok else "No")
col3.metric("Takeoff board speed", f"{mps_to_mph(v_takeoff_mps):.1f} mph" if ok else "—")
col4.metric("Predicted wind floor", f"{wind_floor_mph:.1f} mph" if wind_floor_mph else "—")

# Smart suggestion
suggestion_text, suggested = suggest_adjustment(wind_mps, params)
st.success(suggestion_text)

# Tabs
tab_forces, tab_side, tab_3d, tab_about = st.tabs(["Forces", "Side View", "3D", "About"])

# Forces tab
with tab_forces:
    v = np.linspace(0.1, params["v_max_mps"], 240)
    v_app = apparent_wind_speed(wind_mps, v, params["rel_angle_deg"])
    cl_eff = wing_cl_ramp(v_app, params["wing_cl_max"], params["vaw_roll_in"])
    pump = np.where(v < 3.0, params["skill"]["pump_mult"], 1.0)
    eta_curve = eta_vs_speed(v, params["eta_low_bias"],
                             params["wing_eta_high"] * params["skill"]["eta_mult"],
                             params["eta_half_v"])
    F_thrust, F_vert = wing_forces(v_app, params["wing_area_m2"], cl_eff * pump,
                                   eta_curve, params["vertical_fraction"], params["rho_air"])
    D = water_drag(v, params["rho_water"], params["CdA_taxi"], params["low_speed_taxi"],
                   params["v_hump"], params["sigma"], skill_drag_mult=params["skill"]["drag_mult"])
    L_foil = foil_lift_max(v, params["rho_water"], params["foil_area_m2"],
                           params["foil_cl_max"], params["foil_eff_mult_total"])

    W_eff_arr = np.maximum(params["weight_N"] - F_vert, 0.0)
    vertical_ok = L_foil >= (W_eff_arr * params["lift_safety"])
    horizontal_ok = F_thrust >= (D * params["taxis_margin"])
    both = vertical_ok & horizontal_ok

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mps_to_mph(v), y=L_foil, mode="lines", name="Foil lift (N)"))
    fig.add_trace(go.Scatter(x=mps_to_mph(v), y=W_eff_arr * params["lift_safety"],
                             mode="lines", name="Required lift (N)"))
    fig.add_trace(go.Scatter(x=mps_to_mph(v), y=F_thrust, mode="lines", name="Wing thrust (N)"))
    fig.add_trace(go.Scatter(x=mps_to_mph(v), y=D * params["taxis_margin"], mode="lines", name="Water drag req (N)"))

    if both.any():
        v_to = v[np.argmax(both)]
        fig.add_vline(x=mps_to_mph(v_to), line_dash="dash", annotation_text="Takeoff", annotation_position="top")
    fig.update_layout(xaxis_title="Board speed (mph)", yaxis_title="Force (N)", height=480)
    st.plotly_chart(fig, use_container_width=True)

# Side view
with tab_side:
    v_sel_mph = st.slider("Board speed for side-view (mph)", 1.0, mps_to_mph(params["v_max_mps"]), mps_to_mph(v_takeoff_mps) if ok else 7.0, 0.5)
    v_sel = mph_to_mps(v_sel_mph)
    v_app = apparent_wind_speed(wind_mps, v_sel, params["rel_angle_deg"])
    cl_eff = wing_cl_ramp(v_app, params["wing_cl_max"], params["vaw_roll_in"])
    eta_val = float(eta_vs_speed(np.array([v_sel]), params["eta_low_bias"],
                                 params["wing_eta_high"] * params["skill"]["eta_mult"],
                                 params["eta_half_v"])[0])
    pump = params["skill"]["pump_mult"] if v_sel < 3.0 else 1.0
    F_thrust, F_vert = wing_forces(v_app, params["wing_area_m2"], cl_eff * pump,
                                   eta_val, params["vertical_fraction"], params["rho_air"])
    D = water_drag(np.array([v_sel]), params["rho_water"], params["CdA_taxi"], params["low_speed_taxi"],
                   params["v_hump"], params["sigma"], skill_drag_mult=params["skill"]["drag_mult"])[0]
    L = foil_lift_max(np.array([v_sel]), params["rho_water"], params["foil_area_m2"],
                      params["foil_cl_max"], params["foil_eff_mult_total"])[0]

    # 2D arrows
    fig2 = go.Figure()
    def add_arrow(fig, x0, y0, dx, dy, name):
        fig.add_annotation(x=x0+dx, y=y0+dy, ax=x0, ay=y0, showarrow=True, arrowhead=3, arrowsize=1.2, xref="x", yref="y", text=name)
    add_arrow(fig2, 0, 0, 0, -params["weight_N"], "Weight")
    add_arrow(fig2, 0, 0, 0, L, "Foil lift")
    add_arrow(fig2, 0, 0, F_thrust, 0, "Thrust")
    add_arrow(fig2, 0, 0, -D, 0, "Drag")
    add_arrow(fig2, 0, 0, 0, F_vert, "Wing vertical")
    fig2.update_xaxes(range=[-max(D, F_thrust)*1.2, max(D, F_thrust)*1.2])
    fig2.update_yaxes(range=[-params["weight_N"]*1.2, max(L, F_vert, params["weight_N"])*1.2])
    fig2.update_layout(height=480, xaxis_title="Horizontal (N)", yaxis_title="Vertical (N)")
    st.plotly_chart(fig2, use_container_width=True)

# 3D view (minimal orthographic)
with tab_3d:
    vectors = [
        ("Weight", (0,0,0), (0,0,-params["weight_N"])),
        ("Foil lift", (0,0,0), (0,0,float(L if 'L' in locals() else 0.0))),
        ("Thrust", (0,0,0), (float(F_thrust[0] if isinstance(F_thrust, np.ndarray) else F_thrust),0,0)),
        ("Drag", (0,0,0), (-float(D if isinstance(D, (int,float)) else D),0,0)),
        ("Wing vertical", (0,0,0), (0,0,float(F_vert[0] if isinstance(F_vert, np.ndarray) else F_vert))),
    ]
    fig3 = go.Figure()
    for name, (x0,y0,z0), (dx,dy,dz) in vectors:
        fig3.add_trace(go.Scatter3d(
            x=[x0, x0+dx], y=[y0, y0+dy], z=[z0, z0+dz],
            mode="lines+text",
            text=[None, name],
            textposition="top center"
        ))
    fig3.update_layout(scene=dict(xaxis_title="X (N)", yaxis_title="Y (N)", zaxis_title="Z (N)"),
                       height=520)
    st.plotly_chart(fig3, use_container_width=True)

# About
with tab_about:
    st.markdown("""
**Model overview.** Two gates must be satisfied to fly:
1) **Vertical** — foil lift must exceed effective rider weight (after any vertical assist from the wing) with a safety margin.
2) **Horizontal** — wing thrust must exceed water drag (board/foil taxi + hump) with a safety margin.

**Units:** All UI inputs/outputs are **mph**. Internally, calculations use SI units for consistency.

**Assumptions & dials.**
- ρ_air = 1.225 kg/m³, ρ_water = 1025 kg/m³ (editable).
- Wing CL ramps with apparent wind `v_app` via a smooth exponential roll-in.
- Per-wing CL caps and η_high come from `gear.json` (Flux vs Wasp).
- η increases with board speed (bias + half-speed parameters).
- Low-speed taxi-drag hump (magnitude/center/width) models the takeoff "hump".
- Skill multipliers: η, pumping boost (<3 m/s), and taxi drag.
- Safety margins for lift and thrust are user-tunable.

**Presets.**
Use the buttons below to export/import JSON presets.
    """)

# Preset export/import
preset = dict(
    timestamp=str(pd.Timestamp.utcnow()),
    wind_mph=wind_mph, rel_angle_deg=rel_angle_deg,
    weight_lbs=weight_lbs, skill=skill_name, brand=brand, model=model, wing_size=wing_size,
    front=front, stab=stab, fuse=fuse,
    physics=dict(
        rho_air=rho_air, rho_water=rho_water, vaw_roll_in=vaw_roll_in,
        eta_low_bias=eta_low_bias, eta_half_v=eta_half_v, vertical_fraction=vertical_fraction,
        foil_cl_max=foil_cl_max, CdA_taxi=CdA_taxi, low_speed_taxi=low_speed_taxi,
        v_hump=v_hump, sigma=sigma, lift_safety=lift_safety, taxis_margin=taxis_margin
    )
)
st.download_button("Download current preset", data=json.dumps(preset, indent=2), file_name="wingfoil_preset.json")

uploaded = st.file_uploader("Load preset JSON", type=["json"])
if uploaded is not None:
    try:
        preset_in = json.load(uploaded)
        st.json(preset_in)
        st.info("Preset loaded (values shown above). Apply manually for now; automated apply is a small TODO.")
    except Exception as e:
        st.error(f"Failed to parse preset: {e}")
