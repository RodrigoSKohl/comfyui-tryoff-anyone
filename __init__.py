from .nodes.TryOffAnyoneNode import TryOffAnyoneNode


# Registra o nó no ComfyUI
NODE_CLASS_MAPPINGS = {
    "TryOffAnyoneNode": TryOffAnyoneNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TryOffAnyoneNode": "TryOff Anyone Node",
}


print("\033[34mComfyUI Custom Nodes: \033[92mLoaded TryOff Anyone Node\033[0m")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']