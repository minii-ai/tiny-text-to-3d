from .base import PointCloudDiT
from .class_condition import ClassConditionalPointCloudDiT
from .clip_condition import CLIPConditionalPointCloudDiT
from .super_res import SuperResPointCloudDiT
from .unconditional import UnconditionalPointCloudDiT


def model_from_config(config: dict):
    config = {**config}
    model_type = config["type"]
    del config["type"]

    if model_type == "UnconditionalPointCloudDiT":
        return UnconditionalPointCloudDiT(**config)
    elif model_type == "ClassConditionalPointCloudDiT":
        return ClassConditionalPointCloudDiT(**config)
    elif model_type == "CLIPConditionalPointCloudDiT":
        return CLIPConditionalPointCloudDiT(**config)
    elif model_type == "SuperResPointCloudDiT":
        return SuperResPointCloudDiT(**config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
