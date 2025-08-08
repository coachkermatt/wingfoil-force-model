# Wing Foiling Force Model — AXIS + Ozone (MPH) — Refactored

Modular **Streamlit** app that simulates wing-foiling forces and gives **gear suggestions**.

- **MPH-only UI**; SI for internals.
- **Refactored modules**: physics engine, model, UI, URL state.
- **Calibrated defaults** for 18 mph / Flux 5.0 / Spitfire 840 / 165 lb / Intermediate.
- **Shareable URL + auto-apply**.
- Gear database in `data/gear.json` (add other brands easily).

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Structure
```
app.py
wingfoil_app/
  __init__.py
  config.py        # defaults, skills
  physics.py       # forces & fluids helpers
  model.py         # horizontal/vertical gate logic
  url.py           # shareable URL + presets
  ui.py            # sidebar + charts
  styles.css       # optional styles
data/
  gear.json
.github/workflows/ci.yml
```

