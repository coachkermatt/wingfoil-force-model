
import json
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from .physics import mph_to_mps, mps_to_mph, apparent_wind_speed, wing_cl_ramp, eta_vs_speed, wing_forces, water_drag, foil_lift_max

def sidebar_controls(GEAR, SKILL):
    st.sidebar.header("Inputs")

    st.sidebar.number_input("True wind (mph)", min_value=0.0, step=0.5, key="wind_mph")
    st.sidebar.slider("Relative wind angle (deg)", 30, 150, key="rel_angle_deg", help="90° ~ beam reach; lower values skew upwind.")
    st.sidebar.number_input("Rider weight (lb)", min_value=60.0, step=1.0, key="weight_lbs")
    st.sidebar.selectbox("Skill", list(SKILL.keys()), key="skill_name")

    # Wings
    st.sidebar.subheader("Wing (handheld)")
    brands = list(GEAR["wings"].keys())
    st.session_state.setdefault("wing_brand", brands[0])
    st.sidebar.selectbox("Brand", brands, index=brands.index(st.session_state["wing_brand"]), key="wing_brand")
    wing_models = list(GEAR["wings"][st.session_state["wing_brand"]].keys())
    st.sidebar.selectbox("Model", wing_models, key="wing_model")
    sizes = GEAR["wings"][st.session_state["wing_brand"]][st.session_state["wing_model"]]["sizes_m2"]
    st.sidebar.select_slider("Size (m²)", options=sizes, key="wing_size_m2")
    wing_cl_max = GEAR["wings"][st.session_state["wing_brand"]][st.session_state["wing_model"]]["cl_max"]
    wing_eta_high = GEAR["wings"][st.session_state["wing_brand"]][st.session_state["wing_model"]]["eta_high"]

    # Foils
    st.sidebar.subheader("Foil (AXIS)")
    fronts = GEAR["foils"]["AXIS"]["front_wings"]
    stabs = GEAR["foils"]["AXIS"]["stabilizers"]
    fuses = GEAR["foils"]["AXIS"]["fuselages"]

    front_labels = [f'{fw["model"]} ({fw["area_cm2"]} cm²)' for fw in fronts]
    st.sidebar.selectbox("Front wing", list(range(len(fronts))), format_func=lambda i: front_labels[i], key="front_idx")
    front = fronts[st.session_state["front_idx"]]

    stab_labels = [f'{s["model"]} ({s["area_cm2"]} cm²)' for s in stabs]
    st.sidebar.selectbox("Stabilizer", list(range(len(stabs))), format_func=lambda i: stab_labels[i], key="stab_idx")
    stab = stabs[st.session_state["stab_idx"]]

    fuse_labels = [f'{f["model"]}' for f in fuses]
    st.sidebar.selectbox("Fuselage", list(range(len(fuses))), format_func=lambda i: fuse_labels[i], key="fuse_idx")
    fuse = fuses[st.session_state["fuse_idx"]]

    st.sidebar.subheader("Display & Scan")
    st.sidebar.slider("Max board speed to scan (mph)", 10.0, 40.0, key="v_max_mph")
    st.sidebar.checkbox("Mobile UI (larger touch targets)", value=False, key="mobile_ui")

    with st.sidebar.expander("Physics & Safety (advanced)", expanded=False):
        st.number_input("Air density (kg/m³)", value=st.session_state["rho_air"], step=0.01, min_value=0.8, max_value=1.35, key="rho_air")
        st.number_input("Water density (kg/m³)", value=st.session_state["rho_water"], step=1.0, min_value=990.0, max_value=1050.0, key="rho_water")
        st.number_input("Wing CL roll-in (m/s)", value=st.session_state["vaw_roll_in"], step=0.1, help="Bigger = slower CL ramp.", key="vaw_roll_in")
        st.number_input("η low-speed bias (0–1)", value=st.session_state["eta_low_bias"], step=0.05, min_value=0.0, max_value=1.0, key="eta_low_bias")
        st.number_input("η half-speed (m/s)", value=st.session_state["eta_half_v"], step=0.1, key="eta_half_v")
        st.number_input("Wing vertical fraction (0–0.6)", value=st.session_state["vertical_fraction"], step=0.01, min_value=0.0, max_value=0.6, key="vertical_fraction")
        st.number_input("Foil CL_max", value=st.session_state["foil_cl_max"], step=0.02, min_value=0.3, max_value=1.0, key="foil_cl_max")
        st.number_input("CdA_taxi (m²)", value=st.session_state["CdA_taxi"], step=0.005, min_value=0.005, max_value=0.2, key="CdA_taxi")
        st.number_input("Low-speed taxi hump", value=st.session_state["low_speed_taxi"], step=0.1, min_value=0.0, max_value=3.0, key="low_speed_taxi")
        st.number_input("Hump peak speed (m/s)", value=st.session_state["v_hump"], step=0.1, min_value=0.5, max_value=4.0, key="v_hump")
        st.number_input("Hump width σ (m/s)", value=st.session_state["sigma"], step=0.1, min_value=0.1, max_value=3.0, key="sigma")
        st.number_input("Lift safety margin", value=st.session_state["lift_safety"], step=0.01, min_value=1.0, max_value=1.3, key="lift_safety")
        st.number_input("Taxi thrust margin", value=st.session_state["taxis_margin"], step=0.01, min_value=1.0, max_value=1.5, key="taxis_margin")

    return wing_eta_high, wing_cl_max, fronts, stabs, fuses, front, stab, fuse

def forces_chart(v, v_to, L_foil, W_req, F_thrust, D_req):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v, y=L_foil, mode="lines", name="Foil lift (N)"))
    fig.add_trace(go.Scatter(x=v, y=W_req, mode="lines", name="Required lift (N)"))
    fig.add_trace(go.Scatter(x=v, y=F_thrust, mode="lines", name="Wing thrust (N)"))
    fig.add_trace(go.Scatter(x=v, y=D_req, mode="lines", name="Water drag req (N)"))
    if v_to is not None:
        fig.add_vline(x=v_to, line_dash="dash", annotation_text="Takeoff", annotation_position="top")
    fig.update_layout(xaxis_title="Board speed (mph)", yaxis_title="Force (N)", height=480)
    return fig
