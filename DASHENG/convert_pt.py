import json, torch
from pathlib import Path

def read_config(model_dir):
    with open(Path(model_dir) / "config.json", "r") as f:
        return json.load(f)

def read_state_dict(model_dir):
    from glob import glob
    p = Path(model_dir)
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        # merge safetensors shards
        from safetensors.torch import load_file
        with open(idx, "r") as f:
            index = json.load(f)
        merged = {}
        for shard in {Path(x).name for x in index["weight_map"].values()}:
            merged.update(load_file(str(p / shard), device="cpu"))
        return merged
    st_path = p / "model.safetensors"
    if st_path.exists():
        from safetensors.torch import load_file
        return load_file(str(st_path), device="cpu")
    ptbin = p / "pytorch_model.bin"
    if ptbin.exists():
        return torch.load(str(ptbin), map_location="cpu")
    raise FileNotFoundError("No model.safetensors(.index.json) or pytorch_model.bin found")

def normalize_keys(state_dict):
    def strip_prefix(k):
        for pref in ("model.", "encoder.", "module."):
            if k.startswith(pref):
                return k[len(pref):]
        return k
    return {strip_prefix(k): v for k, v in state_dict.items()}

def map_config_for_dasheng(cfg):
    # Ensure required keys exist; fill/rename as needed for Dasheng constructor
    # Example minimal fields:
    mapped = dict(cfg)
    # If your HF config uses different names, set these explicitly:
    # mapped["embed_dim"] = cfg.get("hidden_size", cfg.get("embed_dim"))
    # mapped["depth"] = cfg.get("num_hidden_layers", cfg.get("depth"))
    # mapped["num_heads"] = cfg.get("num_attention_heads", cfg.get("num_heads"))
    # Also ensure patch settings, pooling, target_length, hop_size match Dasheng
    return mapped

def convert_hf_to_dasheng_pt(model_dir, out_path):
    cfg = map_config_for_dasheng(read_config(model_dir))
    sd = normalize_keys(read_state_dict(model_dir))
    torch.save({"model": sd, "config": cfg}, out_path)
    print(f"Saved Dasheng checkpoint to {out_path}")

# convert_hf_to_dasheng_pt("/path/to/hf_repo_dir", "/path/to/dasheng_converted.pt")