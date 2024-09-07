import requests
import os

_huggingface_model_links = {
    "STU-Net-S": {
        "TotalSegmentator":
        dict(
            repo_id="ziyanhuang/STU-Net",
            filename="small_ep4k.model",
            model_config=dict(
                input_channels=1,
                num_classes=105,
            ),
        ),
    },
    "STU-Net-B": {
        "TotalSegmentator":
        dict(
            repo_id="ziyanhuang/STU-Net",
            filename="base_ep4k.model",
            model_config=dict(
                input_channels=1,
                num_classes=105,
            ),
        ),
        "CT-ORG":
        dict(
            repo_id="blueyo0/STU-Net_CT-ORG",
            filename="base_ep1k.model",
            model_config=dict(
                input_channels=1,
                num_classes=7,
            ),
        ),
        "FeTA21":
        dict(
            repo_id="blueyo0/STU-Net_FeTA21",
            filename="base_ep1k.model",
            model_config=dict(
                input_channels=1,
                num_classes=8,
            ),
        ),
        "BraTS21":
        dict(
            repo_id="blueyo0/STU-Net_BraTS21",
            filename="base_ep1k.model",
            model_config=dict(
                input_channels=4,
                num_classes=5,
            ),
        ),
    },
    "STU-Net-L": {
        "TotalSegmentator":
        dict(
            repo_id="ziyanhuang/STU-Net",
            filename="large_ep4k.model",
            model_config=dict(
                input_channels=1,
                num_classes=105,
            ),
        ),
    },
    "STU-Net-H": {
        "TotalSegmentator":
        dict(
            repo_id="ziyanhuang/STU-Net",
            filename="huge_ep4k.model",
            model_config=dict(
                input_channels=1,
                num_classes=105,
            ),
        ),
    },
}

_HF_ENDPOINT = "https://hf-mirror.com"


def get_huggingface_model_cfg(model_name, pretrained_dataset):
    if (model_name not in _huggingface_model_links):
        raise RuntimeError(
            f"Model {model_name} not found on huggingface, please use the path directly.")
    if (pretrained_dataset not in _huggingface_model_links[model_name]):
        raise RuntimeError(
            f"Pretrained weights for {model_name} on {pretrained_dataset} not found, please check it out."
        )

    return _huggingface_model_links[model_name][pretrained_dataset]


def get_huggingface_model_url(repo_id, fname):
    return f"https://huggingface.co/{repo_id}/blob/main/{fname}"


def is_huggingface_connected():
    url = "https://huggingface.co/models"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print("request fail", e)
        return False


def set_huggingface_endpoint_status(status):
    if (status):
        os.environ["HF_ENDPOINT"] = _HF_ENDPOINT
        print(f"cannot connect to huggingface, try to download from {_HF_ENDPOINT}")
    else:
        if "HF_ENDPOINT" in os.environ:
            del os.environ["HF_ENDPOINT"]


def parse_hf_url(hf_url):
    hf_url = hf_url.replace("https://huggingface.co/", "")
    repo_id, model_path = hf_url.split("/blob/")
    filename = os.path.basename(model_path)
    hf_cfg = dict(
        repo_id=repo_id,
        filename=filename,
    )
    return hf_cfg
