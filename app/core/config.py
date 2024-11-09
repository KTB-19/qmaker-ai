from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "QuestionMakerAI"
    OPENAI_API_KEY: str = ""

    class Config:
        env_file = ".env"


settings = Settings()