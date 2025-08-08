
from dataclasses import dataclass

SKILL = {
    "Beginner":     {"eta_mult": 0.85, "pump_mult": 0.90, "drag_mult": 1.15},
    "Intermediate": {"eta_mult": 1.05, "pump_mult": 1.05, "drag_mult": 1.00},
    "Advanced":     {"eta_mult": 1.05, "pump_mult": 1.05, "drag_mult": 1.00},
    "Expert":       {"eta_mult": 1.10, "pump_mult": 1.10, "drag_mult": 0.95},
}

DEFAULTS = dict(
    # Inputs
    wind_mph=18.0,
    rel_angle_deg=90,
    weight_lbs=165.0,
    skill_name="Intermediate",
    wing_brand="Ozone",
    wing_model="Flux",
    wing_size_m2=5.0,
    foil_brand="AXIS",
    front_idx=1,    # Spitfire 840
    stab_idx=2,     # Prog 400
    fuse_idx=1,     # Standard
    v_max_mph=27.0,
    mobile_ui=False,

    # Physics (calibrated)
    rho_air=1.23,
    rho_water=1025.0,
    vaw_roll_in=2.20,
    eta_low_bias=0.85,
    eta_half_v=2.00,
    vertical_fraction=0.05,
    foil_cl_max=0.66,
    CdA_taxi=0.01,
    low_speed_taxi=0.40,
    v_hump=1.70,
    sigma=0.70,
    lift_safety=1.02,
    taxis_margin=1.05,
)
