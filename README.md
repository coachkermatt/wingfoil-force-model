# Wing Foiling Force Model — AXIS + Ozone (MPH-only)

An interactive **Streamlit app** that simulates wing foiling forces and gives **gear recommendations** based on rider weight, wind, skill, and equipment.  
Includes **AXIS** front wings/stabs/fuselages and **Ozone** Flux/Wasp wings.

> **Units:** The UI uses **MPH only** for wind and board speed. Internally, physics run in SI units.

## Features
- Physics model for vertical (lift) and horizontal (thrust) gates to fly.
- Apparent-wind CL roll-in; efficiency (η) ramps with board speed.
- Low-speed taxi-drag “hump” and skill multipliers (η, pumping, drag).
- Gear database for AXIS & Ozone equipment (editable `gear.json`).
- Smart gear suggestions (wing size up/down, or front wing swap).
- Visualizations: force curves with takeoff marker, side-view vectors, minimal 3D forces.
- Presets: export/import JSON (About tab).

## Install & Run (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run in Colab (Optional)
- Open `notebooks/exploration.ipynb` and run cells.
- A prefilled `.env` with ngrok token is included locally for convenience; **do not commit `.env`**.

## Requirements
- Python 3.9+
- `streamlit`, `plotly`, `numpy`, `pandas`, `python-dotenv`, `pyngrok`

## Environment
- `.env` (local only, ignored by git) contains your ngrok token:
  ```
  NGROK_AUTH_TOKEN=30yy15TAGOEdaIJ6S47LIl3cVb9_7yVeQdfRNEdPL3DxWhXmZ
  ```

## Repo Structure
```
app.py                # Streamlit app (MPH-only UI)
gear.json             # Gear database (AXIS + Ozone)
requirements.txt      # Python dependencies
README.md             # This file
.env                  # Prefilled ngrok token (ignored by git)
.env.example          # Placeholder template
.gitignore
.github/workflows/ci.yml  # GitHub Actions CI workflow
notebooks/
  exploration.ipynb   # Colab helper
```

## Calibration tips
- Tune `vaw_roll_in`, `eta_low_bias`, `eta_half_v`, `CdA_taxi`, `low_speed_taxi`, and safety margins until predicted **wind floor (mph)** matches your threshold sessions within ~1–2 mph.

## License
MIT
