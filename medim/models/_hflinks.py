_huggingface_model_links = {
    "STU-Net-S": {
        "TotalSegmentator": dict(
            repo_id="ziyanhuang/STU-Net",
            filename="small_ep4k.model",
        ),
    },
}

def get_huggingface_model_cfg(model_name, pretrained_dataset):
    if(not model_name in _huggingface_model_links):
        raise RuntimeError(f"Model {model_name} not found on huggingface, please use the path directly.")
    if(not pretrained_dataset in _huggingface_model_links[model_name]):
        raise RuntimeError(f"Pretrained weights for {model_name} on {pretrained_dataset} not found, please check it out.")
    
    return _huggingface_model_links[model_name][pretrained_dataset]
