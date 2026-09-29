from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CITEDESK_", env_file=".env", extra="ignore")

    db_path: str = "citedesk.db"
    ontology_path: str = "data/ontology.toml"
    model: str = "claude-sonnet-5-5"
    top_k: int = 4
    max_chunk_chars: int = 500
    max_output_tokens: int = 600
    # 단가는 공식 가격표를 보고 직접 채운다. 기본 0 이면 비용 칸이 0 으로 찍힐 뿐 추정치를 만들지 않는다.
    usd_per_mtok_in: float = 0.0
    usd_per_mtok_out: float = 0.0
    use_fake_llm: bool = False


def get_settings() -> Settings:
    return Settings()
