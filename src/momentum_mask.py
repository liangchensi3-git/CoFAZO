"""Explicit Qwen parameter masks; no parameter freezing or precision changes."""
import re

MODES=('none','ffn','attention','embedding','attention_embedding')


def resolve_mask(named,mode):
    if mode not in MODES:raise ValueError('Unknown momentum history ablation: '+mode)
    if mode=='none':return [False]*len(named)
    patterns={
        'ffn':r'model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight',
        'attention':r'model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight',
        'embedding':r'model\.embed_tokens\.weight',
        'attention_embedding':r'(model\.embed_tokens\.weight|model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.weight)',
    }
    mask=[re.fullmatch(patterns[mode],n) is not None for n,p in named]
    if not any(mask):raise ValueError('Momentum mask matched no parameters: '+mode)
    if len({id(p) for n,p in named})!=len(named):raise ValueError('Duplicate parameter objects; deduplicate tied weights')
    return mask
