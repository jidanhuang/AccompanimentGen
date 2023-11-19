# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility functions to load from the checkpoints.
Each checkpoint is a torch.saved dict with the following keys:
- 'xp.cfg': the hydra config as dumped during training. This should be used
    to rebuild the object using the audiocraft.models.builders functions,
- 'model_best_state': a readily loadable best state for the model, including
    the conditioner. The model obtained from `xp.cfg` should be compatible
    with this state dict. In the case of a LM, the encodec model would not be
    bundled along but instead provided separately.

Those functions also support loading from a remote location with the Torch Hub API.
They also support overriding some parameters, in particular the device and dtype
of the returned model.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
import typing as tp
import os

from omegaconf import OmegaConf
import torch

from . import builders


HF_MODEL_CHECKPOINTS_MAP = {
    "small": "facebook/musicgen-small",
    "medium": "facebook/musicgen-medium",
    "large": "facebook/musicgen-large",
    "melody": "facebook/musicgen-melody",
}


def _get_state_dict(
    file_or_url_or_id: tp.Union[Path, str],
    filename: tp.Optional[str] = None,
    device='cpu',
    cache_dir: tp.Optional[str] = None,
):
    # Return the state dict either from a file or url
    file_or_url_or_id = str(file_or_url_or_id)
    assert isinstance(file_or_url_or_id, str)

    if os.path.isfile(file_or_url_or_id):
        return torch.load(file_or_url_or_id, map_location=device)

    elif file_or_url_or_id.startswith('https://'):
        return torch.hub.load_state_dict_from_url(file_or_url_or_id, map_location=device, check_hash=True)

    elif file_or_url_or_id in HF_MODEL_CHECKPOINTS_MAP:
        assert filename is not None, "filename needs to be defined if using HF checkpoints"

        repo_id = HF_MODEL_CHECKPOINTS_MAP[file_or_url_or_id]
        file = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        return torch.load(file, map_location=device)

    else:
        raise ValueError(f"{file_or_url_or_id} is not a valid name, path or link that can be loaded.")


def load_compression_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="compression_state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    model = builders.get_compression_model(cfg)
    model.load_state_dict(pkg['best_state'])
    model.eval()
    return model


def load_lm_model_mert(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    del cfg.conditioners['description']
    cfg.conditioners['wav_for_mert']={'model': 'mert', 'mert': {'name': '/data/huangrm/audioLM/bark_trainer_hubert/bark-gui/models/MERT_v1_330M', 'finetune': False}}
    model = builders.get_lm_model_mert(cfg)
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model

def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    # if file_or_url_or_id=='melody':
    #     pkg=torch.load('/data/huangrm/audioLM/musicgen_trainer/models/musicgen-melody/state_dict.bin')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_lm_model(cfg)
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model

def load_lm_model_beatnet(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_lm_model_beatnet(cfg)
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model



def load_lm_model_beatnet_cqtmert(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_lm_model_beatnet_cqtmert(cfg)
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model


def load_lm_model_v2a_melody_mert0_prepend(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    # if file_or_url_or_id=='melody':
    #     pkg=torch.load('/data/huangrm/audioLM/musicgen_trainer/models/musicgen-melody/state_dict.bin')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    cfg.fuser={'cross_attention_pos_emb': False, 'cross_attention_pos_emb_scale': 1, 'sum': [], 'prepend': ['self_wav', 'description'], 'cross': [], 'input_interpolate': []}
    cfg.attribute_dropout={'args': {'active_on_eval': False}, 'text': {}, 'wav': {'self_wav': 0.5}}
    
    cfg.conditioners={'args': {'merge_text_conditions_p': 0.25, 'drop_desc_p': 0.5}, 'self_wav': {'model': 'chroma_stem', 'chroma_stem': {'sample_rate': 32000, 'n_chroma': 12, 'radix2_exp': 14, 'argmax': True, 'cache_path': '/checkpoint/audiocraft/defossez/experiments/magma/chroma_cache_novc', 'match_len_on_eval': False, 'eval_wavs': None, 'n_eval_wavs': 100}}}
    model = builders.get_lm_model_melody_mert0_prepend(cfg)
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model

def load_lm_model_v2a_melody_mert1_prepend(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    # if file_or_url_or_id=='melody':
    #     pkg=torch.load('/data/huangrm/audioLM/musicgen_trainer/models/musicgen-melody/state_dict.bin')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    cfg.fuser={'cross_attention_pos_emb': False, 'cross_attention_pos_emb_scale': 1, 'sum': [], 'prepend': ['self_wav', 'description','vocal_wav_for_mert1'], 'cross': [], 'input_interpolate': []}
    cfg.attribute_dropout={'args': {'active_on_eval': False}, 'text': {}, 'wav': {'self_wav': 0.5}}
    
    cfg.conditioners={'args': {'merge_text_conditions_p': 0.25, 'drop_desc_p': 0.5}, 'self_wav': {'model': 'chroma_stem', 'chroma_stem': {'sample_rate': 32000, 'n_chroma': 12, 'radix2_exp': 14, 'argmax': True, 'cache_path': '/checkpoint/audiocraft/defossez/experiments/magma/chroma_cache_novc', 'match_len_on_eval': False, 'eval_wavs': None, 'n_eval_wavs': 100}}
                      ,'vocal_wav_for_mert1': {'model': 'mert1','mert1': {'name': '/data/huangrm/audioLM/bark_trainer_hubert/bark-gui/models/MERT_v1_330M', 'finetune': False}}}
    model = builders.get_lm_model_melody_mert1_prepend(cfg)
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model
def load_lm_model_v2a_melody(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    # if file_or_url_or_id=='melody':
    #     pkg=torch.load('/data/huangrm/audioLM/musicgen_trainer/models/musicgen-melody/state_dict.bin')
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    cfg.fuser={'cross_attention_pos_emb': False, 'cross_attention_pos_emb_scale': 1, 'sum': [], 'prepend': ['self_wav', 'description'], 'cross': [], 'input_interpolate': []}
    cfg.attribute_dropout={'args': {'active_on_eval': False}, 'text': {}, 'wav': {'self_wav': 0.5}}
    cfg.conditioners={'args': {'merge_text_conditions_p': 0.25, 'drop_desc_p': 0.5}, 'self_wav': {'model': 'chroma_stem', 'chroma_stem': {'sample_rate': 32000, 'n_chroma': 12, 'radix2_exp': 14, 'argmax': True, 'cache_path': '/checkpoint/audiocraft/defossez/experiments/magma/chroma_cache_novc', 'match_len_on_eval': False, 'eval_wavs': None, 'n_eval_wavs': 100}}}
    model = builders.get_lm_model_melody(cfg)
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model

def load_lm_model_5stems(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    model = builders.get_lm_model_5stems(cfg)
    
    model.emb[0].weight.data[:2049, :] =pkg['best_state']['emb.0.weight']
    model.emb[1].weight.data[:2049, :] =pkg['best_state']['emb.1.weight']
    model.emb[2].weight.data[:2049, :] =pkg['best_state']['emb.2.weight']
    model.emb[3].weight.data[:2049, :] =pkg['best_state']['emb.3.weight']
    pkg['best_state'].pop('emb.0.weight')
    pkg['best_state'].pop('emb.1.weight')
    pkg['best_state'].pop('emb.2.weight')
    pkg['best_state'].pop('emb.3.weight')
    
    model.load_state_dict(pkg['best_state'],strict=False)
    model.eval()
    model.cfg = cfg
    return model
def load_lm_model_v2a(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = _get_state_dict(file_or_url_or_id, filename="state_dict.bin", cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    cfg.causal=False
    model = builders.get_lm_model_v2a(cfg)
    # pkg['best_state'].pop('condition_provider.conditioners.description.output_proj.weight', None)
    # pkg['best_state'].pop('condition_provider.conditioners.description.output_proj.bias', None)
    model.load_state_dict(pkg['best_state'],strict=False)
    # model.eval()
    model.cfg = cfg
    return model
