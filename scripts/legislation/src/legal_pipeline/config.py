from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    openstates_api_key: str | None = os.getenv("OPENSTATES_API_KEY")
    legiscan_api_key: str | None = os.getenv("LEGISCAN_API_KEY")
    default_state: str = os.getenv("DEFAULT_STATE", "CO")
    default_session: str | int = os.getenv("DEFAULT_SESSION", "2024")

settings = Settings()
