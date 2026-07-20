from . import samplers
from .trellis_image_to_3d import TrellisImageTo3DPipeline
# trellis_text_to_3d bilerek import edilmiyor: yalnızca open3d gerektiren
# metinden-3D akışı için var, bu adapter yalnızca image-to-3D destekliyor
# (diğer dört aileyle aynı desen) — gereksiz ~500MB'lık bir bağımlılığı
# atlıyor. Metin-tabanlı bir pipeline.json ile from_pretrained() çağrılırsa
# aşağıdaki globals() araması net bir KeyError verir.


def from_pretrained(path: str):
    """
    Load a pipeline from a model folder or a Hugging Face model hub.

    Args:
        path: The path to the model. Can be either local path or a Hugging Face model name.
    """
    import os
    import json
    is_local = os.path.exists(f"{path}/pipeline.json")

    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(path, "pipeline.json")

    with open(config_file, 'r') as f:
        config = json.load(f)
    return globals()[config['name']].from_pretrained(path)
