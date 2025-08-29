"""Global Vars"""

import os


class GlobalVars:
    """
    Store some global runtime constants.
    """

    def __init__(self):

        # Runtime constants: New generation token ratio estimation
        self.default_init_new_token_ratio = float(
            os.environ.get("SGLANG_INIT_NEW_TOKEN_RATIO", 0.7)
        )
        self.default_min_new_token_ratio_factor = float(
            os.environ.get("SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR", 0.14)
        )
        self.default_new_token_ratio_decay_steps = float(
            os.environ.get("SGLANG_NEW_TOKEN_RATIO_DECAY_STEPS", 600)
        )

        # Runtime constants: others
        self.retract_decode_steps = int(
            os.environ.get("SGLANG_RETRACT_DECODE_STEPS", 20)
        )


global_vars = GlobalVars()
