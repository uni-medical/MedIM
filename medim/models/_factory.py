from ._registry import model_entrypoint

def create_model(
    model_name: str,
    pretrained: bool = False,
    checkpoint_path: str = '',
):
   create_fn = model_entrypoint(model_name)
   return create_fn(pretrained=pretrained, 
                    checkpoint_path=checkpoint_path)