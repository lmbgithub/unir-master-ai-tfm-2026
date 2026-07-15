from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    nats_url: str = "nats://nats:4222"
    model_config = SettingsConfigDict(env_file=".env", extra="allow")


class LLMAgentConfig(AgentConfig):
    llm_url: str = "http://llm:11434"
    llm_model: str = "lfm2-700m-q4_k_m.gguf"
    llm_timeout: float = 120.0
    llm_num_ctx: int = 128000
