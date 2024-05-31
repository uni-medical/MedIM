from typing import Any, Callable, Dict

_model_entrypoints: Dict[str, Callable[..., Any]] = {}  # mapping of model names to architecture entrypoint fns

def register_model(model_name):
    def decorator(create_fn):
        _model_entrypoints[model_name] = create_fn
        return create_fn
    return decorator

def model_entrypoint(model_name: str) -> Callable[..., Any]:
    """Fetch a model entrypoint for specified model name
    """
    return _model_entrypoints[model_name]