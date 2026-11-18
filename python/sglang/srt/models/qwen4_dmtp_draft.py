"""Explicit dual-anchor residual export; ordinary DFlash2 is not DMTP."""

import os

from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.speculative.dmtp_qsa_contract import (
    T1Conditioner,
    validate_draft_contract,
)


class Qwen4DMTPT1DraftModel(DFlash2DraftModel):
    def __init__(self, config, quant_config=None, prefix=""):
        if os.getenv("SGLANG_DMTP_T1_RESIDUAL", "0") != "1":
            raise ValueError(
                "This draft requires the opt-in QSA DMTP worker; it cannot use ordinary DFlash2"
            )
        validate_draft_contract(config)
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.mtp_prefix_conditioner = T1Conditioner(
            config.hidden_size, config.rms_norm_eps
        )

    def load_weights(self, weights):
        # Check backbone, conv, selector and residual parameters together.
        # Filtering residual keys first would hide them from the strict loader.
        return super().load_weights(
            (name.removeprefix("model."), value) for name, value in weights
        )


EntryClass = Qwen4DMTPT1DraftModel
