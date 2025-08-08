
import json
import streamlit as st
from urllib.parse import urlencode

def _apply_to_state(d):
    for k, v in d.items():
        if v is not None:
            st.session_state[k] = v

def init_state_from_query(defaults, fronts, stabs, fuses):
    # Populate defaults once
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

    if st.session_state.get('_qs_applied'):
        return

    qs = st.query_params

    def get_str(name): return qs.get(name, None)
    def get_float(name):
        v = get_str(name)
        try: return float(v) if v is not None else None
        except: return None
    def get_int(name):
        v = get_str(name)
        try: return int(float(v)) if v is not None else None
        except: return None
    def get_bool(name):
        v = get_str(name)
        if v is None: return None
        return str(v).lower() in ('1','true','yes','y','on')

    _apply_to_state(dict(
        wind_mph=get_float('wind_mph'),
        rel_angle_deg=get_int('rel_angle_deg'),
        weight_lbs=get_float('weight_lbs'),
        skill_name=get_str('skill'),
        wing_model=get_str('model'),
        wing_size_m2=get_float('wing_size'),
        v_max_mph=get_float('v_max_mph'),
        mobile_ui=get_bool('mobile_ui'),
        rho_air=get_float('rho_air'),
        rho_water=get_float('rho_water'),
        vaw_roll_in=get_float('vaw_roll_in'),
        eta_low_bias=get_float('eta_low_bias'),
        eta_half_v=get_float('eta_half_v'),
        vertical_fraction=get_float('vertical_fraction'),
        foil_cl_max=get_float('foil_cl_max'),
        CdA_taxi=get_float('CdA_taxi'),
        low_speed_taxi=get_float('low_speed_taxi'),
        v_hump=get_float('v_hump'),
        sigma=get_float('sigma'),
        lift_safety=get_float('lift_safety'),
        taxis_margin=get_float('taxis_margin'),
    ))

    fm = get_str('front_model')
    if fm:
        idx = next((i for i, fw in enumerate(fronts) if fw['model'] == fm), None)
        if idx is not None: st.session_state['front_idx'] = idx

    sm = get_str('stab_model')
    if sm:
        idx = next((i for i, s in enumerate(stabs) if s['model'] == sm), None)
        if idx is not None: st.session_state['stab_idx'] = idx

    um = get_str('fuse_model')
    if um:
        idx = next((i for i, f in enumerate(fuses) if f['model'] == um), None)
        if idx is not None: st.session_state['fuse_idx'] = idx

    st.session_state['_qs_applied'] = True

def update_share_url(fronts, stabs, fuses):
    s = st.session_state
    params = dict(
        wind_mph=f"{s['wind_mph']:.2f}",
        rel_angle_deg=str(int(s['rel_angle_deg'])),
        weight_lbs=f"{s['weight_lbs']:.1f}",
        skill=s['skill_name'],
        model=s['wing_model'],
        wing_size=f"{s['wing_size_m2']:.1f}",
        front_model=fronts[s['front_idx']]['model'],
        stab_model=stabs[s['stab_idx']]['model'],
        fuse_model=fuses[s['fuse_idx']]['model'],
        v_max_mph=f"{s['v_max_mph']:.1f}",
        mobile_ui='1' if s['mobile_ui'] else '0',
        rho_air=f"{s['rho_air']:.3f}",
        rho_water=f"{s['rho_water']:.1f}",
        vaw_roll_in=f"{s['vaw_roll_in']:.2f}",
        eta_low_bias=f"{s['eta_low_bias']:.3f}",
        eta_half_v=f"{s['eta_half_v']:.2f}",
        vertical_fraction=f"{s['vertical_fraction']:.3f}",
        foil_cl_max=f"{s['foil_cl_max']:.3f}",
        CdA_taxi=f"{s['CdA_taxi']:.3f}",
        low_speed_taxi=f"{s['low_speed_taxi']:.3f}",
        v_hump=f"{s['v_hump']:.2f}",
        sigma=f"{s['sigma']:.2f}",
        lift_safety=f"{s['lift_safety']:.2f}",
        taxis_margin=f"{s['taxis_margin']:.2f}",
    )
    st.query_params.from_dict(params)
    return '?' + urlencode(params)

def apply_preset_to_state(preset, fronts, stabs, fuses):
    wind_mph = preset.get('wind_mph')
    if wind_mph is None and preset.get('units') == 'mph' and 'wind_in' in preset:
        wind_mph = float(preset['wind_in'])

    d = dict(
        wind_mph=wind_mph,
        rel_angle_deg=preset.get('rel_angle_deg'),
        weight_lbs=preset.get('weight_lbs'),
        skill_name=preset.get('skill'),
        wing_model=preset.get('model'),
        wing_size_m2=preset.get('wing_size'),
    )
    phys = preset.get('physics', {})
    d.update(dict(
        rho_air=phys.get('rho_air'),
        rho_water=phys.get('rho_water'),
        vaw_roll_in=phys.get('vaw_roll_in'),
        eta_low_bias=phys.get('eta_low_bias'),
        eta_half_v=phys.get('eta_half_v'),
        vertical_fraction=phys.get('vertical_fraction'),
        foil_cl_max=phys.get('foil_cl_max'),
        CdA_taxi=phys.get('CdA_taxi'),
        low_speed_taxi=phys.get('low_speed_taxi'),
        v_hump=phys.get('v_hump'),
        sigma=phys.get('sigma'),
        lift_safety=phys.get('lift_safety'),
        taxis_margin=phys.get('taxis_margin'),
    ))
    _apply_to_state(d)

    front = preset.get('front', {})
    if isinstance(front, dict) and 'model' in front:
        idx = next((i for i, fw in enumerate(fronts) if fw['model'] == front['model']), None)
        if idx is not None: st.session_state['front_idx'] = idx
    stab = preset.get('stab', {})
    if isinstance(stab, dict) and 'model' in stab:
        idx = next((i for i, s in enumerate(stabs) if s['model'] == stab['model']), None)
        if idx is not None: st.session_state['stab_idx'] = idx
    fuse = preset.get('fuse', {})
    if isinstance(fuse, dict) and 'model' in fuse:
        idx = next((i for i, f in enumerate(fuses) if f['model'] == f['model']), None)
        if idx is not None: st.session_state['fuse_idx'] = idx
