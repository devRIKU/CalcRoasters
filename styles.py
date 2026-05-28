import toml
import os

def apply_theme(personality: str):
    """
    Updates .streamlit/config.toml with theme settings based on personality.
    """
    
    # Define themes
    themes = {
        "Roaster": {
            "primaryColor": "#ff4444",
            "backgroundColor": "#1a0505",
            "secondaryBackgroundColor": "#2e0a0a",
            "textColor": "#ffcccc",
        },
        "Smart": {
            "primaryColor": "#4299e1",
            "backgroundColor": "#f0f4f8",
            "secondaryBackgroundColor": "#ffffff",
            "textColor": "#1a202c",
        },
        "Debater": {
            "primaryColor": "#d69e2e",
            "backgroundColor": "#2d3748",
            "secondaryBackgroundColor": "#1a202c",
            "textColor": "#e2e8f0",
        },
        "Strategic": {
            "primaryColor": "#10b981",
            "backgroundColor": "#0f172a",
            "secondaryBackgroundColor": "#1e293b",
            "textColor": "#cbd5e1",
        },
        # Hacker terminal: bright green on near-black.
        "Tech Nerd": {
            "primaryColor": "#00ff9c",
            "backgroundColor": "#0a0f0d",
            "secondaryBackgroundColor": "#11181a",
            "textColor": "#b8f5d6",
        },
        # Warm cream + forest green: tea gardens & Sittong forest.
        "Chill Squad": {
            "primaryColor": "#7a9e6e",
            "backgroundColor": "#f5efe2",
            "secondaryBackgroundColor": "#e8e0cc",
            "textColor": "#3d3a2f",
        },
        # Muted, drained-out night-of-homework palette.
        "Exhausted Student": {
            "primaryColor": "#6b6488",
            "backgroundColor": "#1c1a26",
            "secondaryBackgroundColor": "#262332",
            "textColor": "#8e88a3",
        },
    }

    selected_theme = themes.get(personality)
    if not selected_theme:
        return

    config_path = ".streamlit/config.toml"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    # Read existing config
    current_config = {}
    if os.path.exists(config_path):
        try:
            current_config = toml.load(config_path)
        except Exception:
            pass

    # Check if update is needed
    if "theme" not in current_config:
        current_config["theme"] = {}
    
    needs_update = False
    for key, value in selected_theme.items():
        if current_config["theme"].get(key) != value:
            current_config["theme"][key] = value
            needs_update = True

    # Write only if changed to avoid unnecessary reloads
    if needs_update:
        with open(config_path, "w") as f:
            toml.dump(current_config, f)



