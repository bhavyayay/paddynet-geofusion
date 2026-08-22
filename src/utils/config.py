import os
from dotenv import load_dotenv
from sentinelhub import SHConfig

load_dotenv()

SH_CLIENT_ID = os.getenv("SH_CLIENT_ID")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET")
SH_BASE_URL = os.getenv("SH_BASE_URL", "https://sh.dataspace.copernicus.eu")
SH_TOKEN_URL = os.getenv(
    "SH_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)


def validate_config():
    if not SH_CLIENT_ID or not SH_CLIENT_SECRET:
        raise ValueError(
            "Missing SH_CLIENT_ID or SH_CLIENT_SECRET. "
            "Make sure you created a .env file in the project root "
            "with both values filled in."
        )
    print("Config loaded successfully.")
    print("Client ID starts with:", SH_CLIENT_ID[:6] + "...")


def get_sh_config():
    validate_config()
    config = SHConfig()
    config.sh_client_id = SH_CLIENT_ID
    config.sh_client_secret = SH_CLIENT_SECRET
    config.sh_base_url = SH_BASE_URL
    config.sh_token_url = SH_TOKEN_URL
    return config


if __name__ == "__main__":
    cfg = get_sh_config()
    print("SHConfig object created successfully.")
    print("Base URL set to:", cfg.sh_base_url)