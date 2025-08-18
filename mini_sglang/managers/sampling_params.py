from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.0
    max_new_tokens: int = 64
    ignore_eos: bool = False

    eos_token_id: int = -1  # Default value, should be set to the actual EOS token ID

    @classmethod
    def init_from_dict(cls, params: dict):
        return cls(
            temperature=params.get("temperature", cls.temperature),
            max_new_tokens=params.get("max_new_tokens", cls.max_new_tokens),
            ignore_eos=params.get("ignore_eos", cls.ignore_eos),
            eos_token_id=params.get("eos_token_id", cls.eos_token_id),
        )
