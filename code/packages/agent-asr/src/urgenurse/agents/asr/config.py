from urgenurse.agents.agent import LLMAgentConfig


class AsrAgentConfig(LLMAgentConfig):
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
