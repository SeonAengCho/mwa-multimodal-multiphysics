from .AffineUNet1 import AffineUNet1
from .AffineUNet2 import AffineUNet2
from .AffineUNet3 import AffineUNet3
from .ConcatNet import ConcatNet
from .CrossAttenUNet8 import CrossAttenUNet8
from .CrossAttenUNet12 import CrossAttenUNet12
from .CrossAttenUNet16 import CrossAttenUNet16

MODELS = {
    "AffineUNet1": AffineUNet1,
    "AffineUNet2": AffineUNet2,
    "AffineUNet3": AffineUNet3,
    "ConcatNet": ConcatNet,
    "CrossAttenUNet8": CrossAttenUNet8,
    "CrossAttenUNet12": CrossAttenUNet12,
    "CrossAttenUNet16": CrossAttenUNet16,
}
CROSS_ATTEN_MODELS = (CrossAttenUNet8, CrossAttenUNet12, CrossAttenUNet16)


def build_model(spec, mode):
    """Build a model from a config entry, e.g. build_model(config.temp_model, "temp")."""
    options = dict(spec)
    name = options.pop("name")
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}'. Available: {', '.join(MODELS)}")
    return MODELS[name](mode=mode, **options)


def pick_seed(seed, name):
    """Return `seed`, or a random one if it is None; print it so the run can be repeated."""
    import random
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
    print(f"{name}: {seed}  (set {name} = {seed} in config.py to reuse it)")
    return seed


def set_train_seed(seed=None):
    """Seed weight init / shuffling / dropout. None draws a random seed. Returns the seed used."""
    import torch
    seed = pick_seed(seed, "train_seed")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_model_name(model):
    """Name used for the weight file: weight/<model name>.pth."""
    name_parts = [model.__class__.__name__]
    if isinstance(model, CROSS_ATTEN_MODELS):
        name_parts.extend([model.model_type, str(model.proj), model.encoder])
    name_parts.append(model.mode)
    return "_".join(name_parts)
