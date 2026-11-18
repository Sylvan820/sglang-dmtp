from transformers import PretrainedConfig


class Qwen4DMTPDraftConfig(PretrainedConfig):
    """Dense DMTP draft schema; does not inherit the Target's HC/MoE defaults."""

    model_type = "qwen4_dmtp_draft"
