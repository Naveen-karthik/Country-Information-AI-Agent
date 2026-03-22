from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    MISTRAL_MODEL: str = "mistral-small-latest"
    MISTRAL_API_URL: str = "https://api.mistral.ai/v1/chat/completions"
    APP_ENV: str = "development"

    class Config:
        env_file = ".env"


settings = Settings()