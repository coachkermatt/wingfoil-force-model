
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from wingfoil_app.config import SKILL, DEFAULTS
from wingfoil_app.physics import mph_to_mps, mps_to_mph, apparent_wind_speed, wing_cl_ramp, eta_vs_speed, wing_forces, water_drag, foil_lift_max
from wingfoil_app.model import can_fly_for_wind, min_wind_threshold_mph
from wingfoil_app.url import init_state_from_query, update_share_url, apply_preset_to_state
from wingfoil_app.ui import sidebar_controls, forces_chart

st.set_page_config(page_title="Wing Foiling Force Model — AXIS + Ozone (MPH)", layout="wide")

GEAR_PATH = Path(__file__).parent / "data" / "gear.json"
with open(GEAR_PATH, "r") as f:
    GEAR = json.load(f)

# init session state from defaults + URL
fronts = GEAR["foils"]["AXIS"]["front_wings"]
stabs = GEAR["foils"]["AXIS"]["stabilizers"]
fuses = GEAR["foils"]["AXIS"]["fuselages"]
init_state_from_query(DEFAULTS, fronts, stabs, fuses)

wing_eta_high, wing_cl_max, fronts, stabs, fuses, front, stab, fuse = sidebar_controls(GEAR, SKILL)

if st.session_state["mobile_ui"]:
    st.markdown("""
        <style>
        .stSlider > div { padding-top: 0.5rem; padding-bottom: 0.5rem; }
        .stButton > button { padding: 0.8rem 1.2rem; font-size: 1.05rem; }
        </style>
    """, unsafe_allow_html=True)

params = dict(
    rel_angle_deg = float(st.session_state["rel_angle_deg"]),
    v_max_mps = float(mph_to_mps(st.session_state["v_max_mph"])),
    weight_N = float(st.session_state["weight_lbs"] * 0.453592 * 9.81),
    skill = SKILL[st.session_state["skill_name"]],
    wing_area_m2 = float(st.session_state["wing_size_m2"]),
    wing_cl_max = float(wing_cl_max),
    wing_eta_high = float(wing_eta_high),
    vaw_roll_in = float(st.session_state["vaw_roll_in"]),
    eta_low_bias = float(st.session_state["eta_low_bias"]),
    eta_half_v = float(st.session_state["eta_half_v"]),
    vertical_fraction = float(st.session_state["vertical_fraction"]),
    foil_area_m2 = float(front["area_cm2"] / 1e4),
    foil_cl_max = float(st.session_state["foil_cl_max"]),
    foil_eff_mult_total = float(front.get("eff_mult", 1.0) * stab.get("eff_mult", 1.0) * fuse.get("eff_mult", 1.0)),
    CdA_taxi = float(st.session_state["CdA_taxi"]),
    low_speed_taxi = float(st.session_state["low_speed_taxi"]),
    v_hump = float(st.session_state["v_hump"]),
    sigma = float(st.session_state["sigma"]),
    lift_safety = float(st.session_state["lift_safety"]),
    taxis_margin = float(st.session_state["taxis_margin"]),
    rho_air = float(st.session_state["rho_air"]),
    rho_water = float(st.session_state["rho_water"]),
)

wind_mps = mph_to_mps(st.session_state["wind_mph"])

st.title("Wing Foiling Force Model — AXIS + Ozone (MPH)")

# KPI cards
ok, v_takeoff_mps, v_eq_mps = can_fly_for_wind(wind_mps, params)
wind_floor_mph = min_wind_threshold_mph(params)
c1, c2, c3, c4 = st.columns(4)
c1.metric("True wind", f"{st.session_state['wind_mph']:.1f} mph")
c2.metric("Takeoff possible", "Yes" if ok else "No")
c3.metric("Takeoff board speed", f"{mps_to_mph(v_takeoff_mps):.1f} mph" if ok else "—")
c4.metric("Predicted wind floor", f"{wind_floor_mph:.1f} mph" if wind_floor_mph else "—")

# Smart suggestion (very light)
if ok:
    st.success("You can downsize to ~4.3 m² and still fly." if st.session_state["wing_size_m2"] > 4.3 else "Your gear is well-matched.")
else:
    st.info("Consider both a larger wing and a larger front wing.")

# Tabs
tab_forces, tab_side, tab_about = st.tabs(["Forces", "Side View", "About / Share"])

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
    both = (L_foil >= (W_eff_arr * params["lift_safety"])) & (F_thrust >= (D * params["taxis_margin"]))
    v_to = mps_to_mph(v[np.argmax(both)]) if both.any() else None

    fig = forces_chart(mps_to_mph(v), v_to, L_foil, W_eff_arr * params["lift_safety"], F_thrust, D * params["taxis_margin"])
    st.plotly_chart(fig, use_container_width=True)

with tab_side:
    default_v = mps_to_mph(v_takeoff_mps) if ok and v_takeoff_mps else 10.0
    st.slider("Board speed for side-view (mph)", 1.0, mps_to_mph(params["v_max_mps"]), default_v, 0.5, key="v_sel_mph")
    v_sel = mph_to_mps(st.session_state["v_sel_mph"])
    v_app = apparent_wind_speed(wind_mps, v_sel, params["rel_angle_deg"])
    cl_eff = wing_cl_ramp(v_app, params["wing_cl_max"], params["vaw_roll_in"])
    pump = params["skill"]["pump_mult"] if v_sel < 3.0 else 1.0
    eta_val = float(eta_vs_speed(np.array([v_sel]), params["eta_low_bias"],
                                 params["wing_eta_high"] * params["skill"]["eta_mult"],
                                 params["eta_half_v"])[0])
    F_thrust, F_vert = wing_forces(v_app, params["wing_area_m2"], cl_eff * pump,
                                   eta_val, params["vertical_fraction"], params["rho_air"])
    D = water_drag(np.array([v_sel]), params["rho_water"], params["CdA_taxi"], params["low_speed_taxi"],
                   params["v_hump"], params["sigma"], skill_drag_mult=params["skill"]["drag_mult"])[0]
    L = foil_lift_max(np.array([v_sel]), params["rho_water"], params["foil_area_m2"],
                      params["foil_cl_max"], params["foil_eff_mult_total"])[0]

    st.write(f"At **{st.session_state['v_sel_mph']:.1f} mph**: Thrust={F_thrust:.0f} N, Drag={D:.0f} N, Lift={L:.0f} N, Wing vertical={float(F_vert):.0f} N")

with tab_about:
    st.markdown("""
**Model overview.** Two gates must be satisfied to fly:
1) **Vertical** — foil lift must exceed effective rider weight (after any vertical assist from the wing) with a safety margin.
2) **Horizontal** — wing thrust must exceed water drag (board/foil taxi + hump) with a safety margin.

**Units:** UI uses **mph**; internal math uses SI.

**Shareable URL:** Click **Update shareable URL** to encode your current setup into the page URL. Opening that URL auto-applies the settings.
""")
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Update shareable URL"):
            update_share_url(fronts, stabs, fuses)
            st.success("URL updated — copy it from the address bar and share.")
    with c2:
        if st.button("Clear URL parameters"):
            st.query_params.clear()
            st.success("Cleared — URL has no parameters now.")
    with c3:
        st.caption("Upload a preset JSON to apply it instantly.")

    preset = dict(
        timestamp=str(pd.Timestamp.utcnow()),
        wind_mph=st.session_state["wind_mph"],
        rel_angle_deg=st.session_state["rel_angle_deg"],
        weight_lbs=st.session_state["weight_lbs"],
        skill=st.session_state["skill_name"],
        brand=st.session_state["wing_brand"],
        model=st.session_state["wing_model"],
        wing_size=st.session_state["wing_size_m2"],
        front=front, stab=stab, fuse=fuse,
        physics=dict(
            rho_air=st.session_state["rho_air"], rho_water=st.session_state["rho_water"], vaw_roll_in=st.session_state["vaw_roll_in"],
            eta_low_bias=st.session_state["eta_low_bias"], eta_half_v=st.session_state["eta_half_v"],
            vertical_fraction=st.session_state["vertical_fraction"],
            foil_cl_max=st.session_state["foil_cl_max"], CdA_taxi=st.session_state["CdA_taxi"], low_speed_taxi=st.session_state["low_speed_taxi"],
            v_hump=st.session_state["v_hump"], sigma=st.session_state["sigma"], lift_safety=st.session_state["lift_safety"], taxis_margin=st.session_state["taxis_margin"]
        )
    )
    st.download_button("Download current preset", data=json.dumps(preset, indent=2), file_name="wingfoil_preset.json")

    uploaded = st.file_uploader("Load preset JSON", type=["json"])
    if uploaded is not None:
        try:
            preset_in = json.load(uploaded)
            apply_preset_to_state(preset_in, fronts, stabs, fuses)
            st.success("Preset applied from file. URL updated.")
            update_share_url(fronts, stabs, fuses)
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Failed to parse/apply preset: {e}")
