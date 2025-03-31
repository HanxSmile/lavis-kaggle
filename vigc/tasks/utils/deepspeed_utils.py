from contextlib import contextmanager
from packaging import version
import deepspeed
import itertools
from deepspeed.runtime.engine import DeepSpeedEngine


def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


def remove_hooks(model: "DeepSpeedEngine") -> None:
    """Removes the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def add_hooks(model: "DeepSpeedEngine") -> None:
    """Adds the optimizer hooks from a DeepSpeed ZeRO-3 model."""
    if not hasattr(model, "optimizer"):  # before the first training step, the model has no optimizer
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError("The model optimizer is None, which is not yet supported.")
    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):
        # Account for renaming in https://github.com/deepspeedai/DeepSpeed/pull/6847
        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
def unwrap_model_for_generation(
        model,
        gather_deepspeed3_params: bool = True,
):
    """
    Context manager to unwrap distributed or accelerated models for generation tasks.

    Args:
        model:
            Model to be unwrapped.
        gather_deepspeed3_params (`bool`, *optional*, defaults to `True`):
            Whether to gather weights for DeepSpeed ZeRO Stage 3 models. If `False`, skips parameter gathering, which
            can be more memory-efficient but may lead to slower generation times.

    Yields:
        Unwrapped model.

    Example:
    ```python
    with unwrap_model_for_generation(model, accelerator) as unwrapped_model:
        generated_outputs = unwrapped_model.generate(input_ids)
    ```
    """

    def _unwrap_model(_model):
        if hasattr(_model, "module"):
            _unwrapped_model = _model.module
        else:
            _unwrapped_model = _model
        return _unwrapped_model

    if not gather_deepspeed3_params:
        yield _unwrap_model(model)
    else:
        with deepspeed.zero.GatheredParameters(model.parameters()):
            remove_hooks(model)
            yield _unwrap_model(model)
            add_hooks(model)
