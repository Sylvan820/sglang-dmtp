"""Explicit dual-anchor residual export; ordinary DFlash2 is not DMTP."""

import os

from sglang.srt.models.dflash import DFlash2DraftModel
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.speculative.dmtp_qsa_contract import T1Conditioner, validate_draft_contract


class Qwen4DMTPT1DraftModel(DFlash2DraftModel):
    def __init__(self, config, quant_config=None, prefix=""):
        if os.getenv("SGLANG_DMTP_QSA", "0") != "1":
            raise ValueError("This draft requires the opt-in QSA DMTP worker; it cannot use ordinary DFlash2")
        validate_draft_contract(config)
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.mtp_prefix_conditioner = T1Conditioner(config.hidden_size, config.rms_norm_eps)

    def load_weights(self, weights):
        seen = set()
        parameters = dict(self.mtp_prefix_conditioner.named_parameters())

        def checked():
            for name, value in weights:
                key = name.removeprefix("model.")
                if key.startswith("mtp_prefix_conditioner."):
                    short = key.removeprefix("mtp_prefix_conditioner.")
                    if short not in parameters or tuple(value.shape) != tuple(parameters[short].shape):
                        raise ValueError(f"Unexpected T1 residual weight/shape: {name}")
                    if key in seen:
                        raise ValueError(f"Duplicate T1 residual weight: {name}")
                    default_weight_loader(parameters[short], value)
                    seen.add(key)
                else:
                    yield name, value

        result = super().load_weights(checked())
        required = {"mtp_prefix_conditioner." + name for name, _ in self.mtp_prefix_conditioner.named_parameters()}
        if not required <= seen:
            raise ValueError(f"Missing trained T1 residual weights: {sorted(required - seen)}")
        return result


EntryClass = Qwen4DMTPT1DraftModel
