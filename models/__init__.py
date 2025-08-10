# models/__init__.py
from .cardamage import DeepLab_V3_Plus_Effi_U_Trans2

def get_model(name, **kwargs):
    if name == "DeepLab_V3_Plus_Effi_USE_Trans2":
        return DeepLab_V3_Plus_Effi_U_Trans2(
            in_channels=kwargs.get("in_channels", 3),
            num_classes=kwargs.get("num_classes", 1)
        )
    else:
        raise ValueError(f"Model {name} not recognized.")