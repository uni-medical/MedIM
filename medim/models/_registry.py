import os
from typing import Any, Callable, Dict
import os.path as osp
import appdirs

_model_entrypoints: Dict[str, Callable[..., Any]] = {
}  # mapping of model names to architecture entrypoint fns
_ckpt_root_dir = os.environ["MEDIM_CKPT_DIR"] if (
    "MEDIM_CKPT_DIR" in os.environ) else appdirs.user_data_dir("checkpoint", "medim")


def register_model(model_name):

    def decorator(create_fn):
        _model_entrypoints[model_name] = create_fn
        return create_fn

    return decorator


def model_entrypoint(model_name: str) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]


def get_pretrained_weights_path(model_name: str, pretrained_dataset: str = 'None'):
    """Fetch the path of the pretrained weights for specified model name
    """
    return osp.join(_ckpt_root_dir, pretrained_dataset, f"{model_name}.pth")


def get_pretrained_weights_path_for_hf(repo_id: str, filename: str = 'None'):
    """Fetch the path of the pretrained weights for specified model name
    """
    return osp.join(_ckpt_root_dir, "huggingface_hub", repo_id, filename)
