import sys
sys.path.append("base_tokenizers/")

from tokenizer_wrapper import VidTokVAEWrapper, WFVAEWrapper, CogvideoXVAEWrapper

def load_vae(vae_name, model_path, embed_dim):
    VAEWrapper = None
    if vae_name == 'wfvae':
        VAEWrapper = WFVAEWrapper
    elif vae_name == 'cogvideox':
        VAEWrapper = CogvideoXVAEWrapper
    if vae_name == 'vidtok':
        VAEWrapper = VidTokVAEWrapper

    if VAEWrapper:
        return VAEWrapper(model_path=model_path, embed_dim=embed_dim)
    else:
        raise Exception(f"Unrecognized vae type: {vae_name}")