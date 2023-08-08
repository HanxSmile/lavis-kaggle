mini_gpt_13b = {
    "source": "/mnt/petrelfs/share_data/zhonghuaping.p/pretrained_weights/pretrained_minigpt4.pth",
    "target": "./minigpt4_proj_13b.pth"
}
mini_gpt_7b = {
    "source": "/mnt/petrelfs/share_data/zhonghuaping.p/pretrained_weights/pretrained_minigpt4_7b_stage1.pth",
    "target": "./minigpt4_proj_7b.pth"
}

import torch
from collections import OrderedDict


def transform(key):
    return key.replace("llama", "llm")


def rename_state_dict_keys(source, key_transformation=transform, target=None):
    """
    source             -> Path to the saved state dict.
    key_transformation -> Function that accepts the old key names of the state
                          dict as the only argument and returns the new key name.
    target (optional)  -> Path at which the new state dict should be saved
                          (defaults to `source`)
    Example:
    Rename the key `layer.0.weight` `layer.1.weight` and keep the names of all
    other keys.
    ```py
    def key_transformation(old_key):
        if old_key == "layer.0.weight":
            return "layer.1.weight"
        return old_key
    rename_state_dict_keys(state_dict_path, key_transformation)
    ```
    """
    if target is None:
        target = source

    state_dict = torch.load(source, map_location="cpu")["model"]
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        new_key = key_transformation(key)
        new_state_dict[new_key] = value

    torch.save(new_state_dict, target)


if __name__ == '__main__':
    rename_state_dict_keys(**mini_gpt_7b)
    rename_state_dict_keys(**mini_gpt_13b)
