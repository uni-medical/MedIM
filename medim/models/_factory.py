from ._registry import model_entrypoint
from ._hf_utils import get_huggingface_model_cfg, get_huggingface_model_url


def create_model(
    model_name: str,
    dataset: str = None,
    pretrained: bool = False,
    checkpoint_path: str = '',
    **kwargs,
):
    create_fn = model_entrypoint(model_name)
    print(f"creating model {model_name}")
    if (dataset):
        print("try to load pretrained weights for", dataset)
        pretrained = True
        hf_cfg = get_huggingface_model_cfg(model_name, dataset)
        checkpoint_path = get_huggingface_model_url(hf_cfg["repo_id"], hf_cfg["filename"])
        kwargs = hf_cfg["model_config"]
    elif (pretrained):
        print("try to load pretrained weights from", checkpoint_path)
    return create_fn(pretrained=pretrained, checkpoint_path=checkpoint_path, **kwargs)
