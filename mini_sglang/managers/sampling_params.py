from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    eos_token_id: int = -1  # Default value, should be set to the actual EOS token ID
