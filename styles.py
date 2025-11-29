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



