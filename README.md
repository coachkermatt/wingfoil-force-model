# Wing Foiling Force Model — AXIS + Ozone (MPH-only)

An interactive **Streamlit app** that simulates wing foiling forces and gives **gear recommendations** based on rider weight, wind, skill, and equipment.  
Includes **AXIS** front wings/stabs/fuselages and **Ozone** Flux/Wasp wings.

> **Units:** The UI uses **MPH only**.

## New in this build
- **Shareable link:** Click **Update shareable URL** (About tab) to encode all settings into the URL.
- **Auto-apply from URL:** Opening a URL with parameters **applies settings automatically**.
- **Preset Apply:** Uploading a preset JSON **immediately applies** it and updates the URL.

## Install & Run (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repo Structure
```
app.py
gear.json
requirements.txt
README.md
.github/workflows/ci.yml
```

## License
MIT
