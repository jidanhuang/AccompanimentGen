# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Main model for using MusicGen. This will combine all the required components
and provide easy access to the generation API.
"""
import tqdm
import torchaudio
import os
import typing as tp
import numpy as np
import torch
import torchaudio.transforms as T

from .encodec import CompressionModel
from .lm import LMModel
from .builders import get_debug_compression_model, get_debug_lm_model
from .loaders import load_compression_model, load_lm_model, load_lm_model_beatnet,load_lm_model_beatnet_cqtmert,HF_MODEL_CHECKPOINTS_MAP,load_lm_model_v2a,load_lm_model_mert,load_lm_model_5stems,load_lm_model_v2a_melody,load_lm_model_v2a_melody_mert0_prepend,load_lm_model_v2a_melody_mert1_prepend
from ..data.audio_utils import convert_audio
from ..modules.conditioners import ConditioningAttributes, WavCondition,ConditioningAttributes_mert1,ConditioningAttributes_mert
from ..utils.autocast import TorchAutocast


MelodyList = tp.List[tp.Optional[torch.Tensor]]
MelodyType = tp.Union[torch.Tensor, MelodyList]


class MusicGen:
    """MusicGen main model with convenient generation API.

    Args:
        name (str): name of the model.
        compression_model (CompressionModel): Compression model
            used to map audio to invertible discrete representations.
        lm (LMModel): Language model over discrete representations.
    """
    def __init__(self, name: str, compression_model: CompressionModel, lm: LMModel,
                 max_duration: float = 30):
        self.name = name
        self.compression_model = compression_model
        self.lm = lm
        self.max_duration = max_duration
        self.device = next(iter(lm.parameters())).device
        self.generation_params: dict = {}
        self.set_generation_params(duration=15)  # 15 seconds by default
        self._progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None
        if self.device.type == 'cpu':
            self.autocast = TorchAutocast(enabled=False)
        else:
            self.autocast = TorchAutocast(
                enabled=True, device_type=self.device.type, dtype=torch.float16)
        
    @property
    def frame_rate(self) -> int:
        """Roughly the number of AR steps per seconds."""
        return self.compression_model.frame_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate of the generated audio."""
        return self.compression_model.sample_rate

    @property
    def audio_channels(self) -> int:
        """Audio channels of the generated audio."""
        return self.compression_model.channels

    @staticmethod
    def get_pretrained_mert(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model_mert(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)
    
    @staticmethod
    def get_pretrained_v2a_melody_mert1_prepend(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model_v2a_melody_mert1_prepend(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)
    @staticmethod
    def get_pretrained_v2a_melody_mert0_prepend(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model_v2a_melody_mert0_prepend(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)
    @staticmethod
    def get_pretrained_v2a_melody(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model_v2a_melody(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)
    
    @staticmethod
    def get_pretrained_beatnet(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model_beatnet(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)

    @staticmethod
    def get_pretrained_beatnet_cqtmert(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model_beatnet_cqtmert(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)

    @staticmethod
    def get_pretrained(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)
    @staticmethod
    def get_pretrained_5stems(name: str = 'melody', device=None):
        """Return pretrained model, we provide four models:
        - small (300M), text to music, # see: https://huggingface.co/facebook/musicgen-small
        - medium (1.5B), text to music, # see: https://huggingface.co/facebook/musicgen-medium
        - melody (1.5B) text to music and text+melody to music, # see: https://huggingface.co/facebook/musicgen-melody
        - large (3.3B), text to music, # see: https://huggingface.co/facebook/musicgen-large
        """

        if device is None:
            if torch.cuda.device_count():
                device = 'cuda'
            else:
                device = 'cpu'

        if name == 'debug':
            # used only for unit tests
            compression_model = get_debug_compression_model(device)
            lm = get_debug_lm_model(device)
            return MusicGen(name, compression_model, lm)

        if name not in HF_MODEL_CHECKPOINTS_MAP:
            raise ValueError(
                f"{name} is not a valid checkpoint name. "
                f"Choose one of {', '.join(HF_MODEL_CHECKPOINTS_MAP.keys())}"
            )

        cache_dir = os.environ.get('MUSICGEN_ROOT', None)
        compression_model = load_compression_model(name, device=device, cache_dir=cache_dir)
        lm = load_lm_model_5stems(name, device=device, cache_dir=cache_dir)
        if name == 'melody':
            lm.condition_provider.conditioners['self_wav'].match_len_on_eval = True

        return MusicGen(name, compression_model, lm)

    def set_generation_params(self, use_sampling: bool = True, top_k: int = 250,
                              top_p: float = 0.0, temperature: float = 1.0,
                              duration: float = 30.0, cfg_coef: float = 3.0,
                              two_step_cfg: bool = False, extend_stride: float = 3):
        """Set the generation parameters for MusicGen.

        Args:
            use_sampling (bool, optional): Use sampling if True, else do argmax decoding. Defaults to True.
            top_k (int, optional): top_k used for sampling. Defaults to 250.
            top_p (float, optional): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
            temperature (float, optional): Softmax temperature parameter. Defaults to 1.0.
            duration (float, optional): Duration of the generated waveform. Defaults to 30.0.
            cfg_coef (float, optional): Coefficient used for classifier free guidance. Defaults to 3.0.
            two_step_cfg (bool, optional): If True, performs 2 forward for Classifier Free Guidance,
                instead of batching together the two. This has some impact on how things
                are padded but seems to have little impact in practice.
            extend_stride: when doing extended generation (i.e. more than 30 seconds), by how much
                should we extend the audio each time. Larger values will mean less context is
                preserved, and shorter value will require extra computations.
        """
        assert extend_stride < self.max_duration, "Cannot stride by more than max generation duration."
        self.extend_stride = extend_stride
        self.duration = duration
        self.generation_params = {
            'use_sampling': use_sampling,
            'temp': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'cfg_coef': cfg_coef,
            'two_step_cfg': two_step_cfg,
        }

    def set_custom_progress_callback(self, progress_callback: tp.Optional[tp.Callable[[int, int], None]] = None):
        """Override the default progress callback."""
        self._progress_callback = progress_callback

    def generate_unconditional(self, num_samples: int, progress: bool = False) -> torch.Tensor:
        """Generate samples in an unconditional manner.

        Args:
            num_samples (int): Number of samples to be generated.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        descriptions: tp.List[tp.Optional[str]] = [None] * num_samples
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        return self._generate_tokens(attributes, prompt_tokens, progress)
    
    def preprocess_audio(self,audio_path ):
        wav, sr = torchaudio.load(audio_path)
        # wav=wav.cpu()
        # wav=wav.to(torch.float16) 
        wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.mean(dim=0, keepdim=True)
        # end_sample = int(self.sample_rate * duration)
        # if wav.shape[1] > self.sample_rate * duration:
        #     wav = wav[:, :end_sample] 
        mask= torch.ones_like(wav, dtype=torch.bool)
        assert wav.shape[0] == 1
        # assert wav.shape[1] <= self.sample_rate * duration

        wav = wav
        wav = wav.unsqueeze(1)
        return wav,mask
    def wav_to_codes(self,wav):
        with torch.no_grad():
            gen_audio = self.compression_model.encode(wav)

        codes, scale = gen_audio

        assert scale is None

        return codes

    def path2codes(self,audiopath:str,addnoise=False):
        vocal,pad_mask = self.preprocess_audio(audiopath) #gen_code: torch.Size([1, 4, 1500]) returns tensor,B,K,T
        noise = torch.randn(vocal.shape)
        # 调整白噪音的幅度（可以根据需要调整）
        noise_amplitude = 0.01#random.uniform(0.05, 0.3) # 调整白噪音的幅度，白噪音会去除冗余局部人声信息，增加宏观结构信息的占比，增加生成的多样性
        vocal = vocal + noise_amplitude * noise
        vocal=self.wav_to_codes(vocal.cuda())
        return vocal
    def get_mid_pad_token(self,vocal):
        b,K,T=vocal.shape
        pad_shape=[b,K,1]
        pad_token=torch.zeros(pad_shape).to(torch.int64)#B,K,K
        # mask_token=torch.zeros(pad_shape).to(torch.bool)#B,K,K
        pad_token[...]=2048
        # mask_token[...]=False
        
        mid_pad_token=pad_token.repeat(1,1,vocal.shape[-2]).to(vocal.device)
        return mid_pad_token
    def get_pad_token(self,vocal,audio_type):
        b,K,T=vocal.shape

        if audio_type=='vocal':
            pad_value=2049
            pad_shape=[b,K,300-T] 
        elif audio_type=='drums':
            pad_value=2051
            pad_shape=[b,K,604-T] 
        elif audio_type=='bass':
            pad_value=2053
            pad_shape=[b,K,908-T] 
        elif audio_type=='piano':
            pad_value=2055
            pad_shape=[b,K,1212-T] 
        
        pad_token=torch.zeros(pad_shape).to(torch.int64)#B,K,K
        # mask_token=torch.zeros(pad_shape).to(torch.bool)#B,K,K
        pad_token[...]=pad_value
        # mask_token[...]=False
        
        # mid_pad_token=pad_token.repeat(1,1,vocal.shape[-2]).to(vocal.device)
        return pad_token
    def get_infer_token(self,vocal,audio_type):
        if audio_type=='drums':
            pad_value=2048
        elif audio_type=='bass':
            pad_value=2050
        elif audio_type=='piano':
            pad_value=2052
        elif audio_type=='other':
            pad_value=2054
        
        b,K,T=vocal.shape
        pad_shape=[b,K,K]
        pad_token=torch.zeros(pad_shape).to(torch.int64)#B,K,K
        # mask_token=torch.zeros(pad_shape).to(torch.bool)#B,K,K
        pad_token[...]=pad_value
        # mask_token[...]=False
        
        # mid_pad_token=pad_token.repeat(1,1,vocal.shape[-2]).to(vocal.device)
        return pad_token    
    def generate(self, descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        return self._generate_tokens(attributes, prompt_tokens, progress)
    
    def generate_v2a(self, vocal_path,descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        codes=self.path2codes(vocal_path)#1,4,300
        mid_pad_token=self.get_mid_pad_token(codes)
        return self._generate_tokens_v2a(codes,mid_pad_token,attributes, prompt_tokens, progress)

    def generate_v2a_pad6s(self, vocal_path,descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        codes=self.path2codes(vocal_path)#1,4,300
        mid_pad_token=self.get_mid_pad_token(codes)
        return self._generate_tokens_v2a_pad6s(codes,mid_pad_token,attributes, prompt_tokens, progress)
    def generate_v2a_pad6s_beatnet(self, vocal_path,descriptions: tp.List[str], progress: bool = False,addnoise=False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        codes=self.path2codes(vocal_path,addnoise=addnoise)#1,4,300
        mid_pad_token=self.get_mid_pad_token(codes)
        return self._generate_tokens_v2a_pad6s_beatnet(codes,mid_pad_token,attributes, prompt_tokens, progress)

    def generate_v2a_pad6s_5stems(self, vocal_path,descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        codes=self.path2codes(vocal_path)#1,4,300
        return self._generate_tokens_v2a_pad6s_5stems(codes,attributes, prompt_tokens, progress)
    def generate_v2a_pad6s_va_a(self, vocal_path,descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
        assert prompt_tokens is None
        codes=self.path2codes(vocal_path)#1,4,300
        return self._generate_tokens_v2a_pad6s_va_a(codes,attributes, prompt_tokens, progress)
    # def generate_v2a_pad6s(self, vocal_path,descriptions: tp.List[str], progress: bool = False) -> torch.Tensor:
    #     """Generate samples conditioned on text.

    #     Args:
    #         descriptions (tp.List[str]): A list of strings used as text conditioning.
    #         progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
    #     """
    #     attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, None)
    #     assert prompt_tokens is None
    #     codes=self.path2codes(vocal_path)#1,4,300
    #     mid_pad_token=self.get_mid_pad_token(codes)
    #     return self._generate_tokens_v2a_pad6s(codes,mid_pad_token,attributes, prompt_tokens, progress)
    def generate_with_chroma_v2a(self, vocal_path,descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)#sr->32000,1channel
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        codes=self.path2codes(vocal_path)#1,4,300
        mid_pad_token=self.get_mid_pad_token(codes)
        return self._generate_tokens_v2a_pad6s(codes,mid_pad_token,attributes, prompt_tokens, progress)
    def generate_with_chroma(self, descriptions: tp.List[str], melody_wavs: MelodyType,
                             melody_sample_rate: int, progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on text and melody.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            melody_wavs: (torch.Tensor or list of Tensor): A batch of waveforms used as
                melody conditioning. Should have shape [B, C, T] with B matching the description length,
                C=1 or 2. It can be [C, T] if there is a single description. It can also be
                a list of [C, T] tensors.
            melody_sample_rate: (int): Sample rate of the melody waveforms.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if isinstance(melody_wavs, torch.Tensor):
            if melody_wavs.dim() == 2:
                melody_wavs = melody_wavs[None]
            if melody_wavs.dim() != 3:
                raise ValueError("Melody wavs should have a shape [B, C, T].")
            melody_wavs = list(melody_wavs)
        else:
            for melody in melody_wavs:
                if melody is not None:
                    assert melody.dim() == 2, "One melody in the list has the wrong number of dims."

        melody_wavs = [
            convert_audio(wav, melody_sample_rate, self.sample_rate, self.audio_channels)#sr->32000,1channel
            if wav is not None else None
            for wav in melody_wavs]
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions=descriptions, prompt=None,
                                                                        melody_wavs=melody_wavs)
        assert prompt_tokens is None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    def generate_continuation(self, prompt: torch.Tensor, prompt_sample_rate: int,
                              descriptions: tp.Optional[tp.List[tp.Optional[str]]] = None,
                              progress: bool = False) -> torch.Tensor:
        """Generate samples conditioned on audio prompts.

        Args:
            prompt (torch.Tensor): A batch of waveforms used for continuation.
                Prompt should be [B, C, T], or [C, T] if only one sample is generated.
            prompt_sample_rate (int): Sampling rate of the given audio waveforms.
            descriptions (tp.List[str], optional): A list of strings used as text conditioning. Defaults to None.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        """
        if prompt.dim() == 2:
            prompt = prompt[None]
        if prompt.dim() != 3:
            raise ValueError("prompt should have 3 dimensions: [B, C, T] (C = 1).")
        prompt = convert_audio(prompt, prompt_sample_rate, self.sample_rate, self.audio_channels)
        if descriptions is None:
            descriptions = [None] * len(prompt)
        attributes, prompt_tokens = self._prepare_tokens_and_attributes(descriptions, prompt)
        assert prompt_tokens is not None
        return self._generate_tokens(attributes, prompt_tokens, progress)

    @torch.no_grad()
    def _prepare_tokens_and_attributes(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (tp.Optional[torch.Tensor], optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes(text={'description': description})
            for description in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path='null_wav')  # type: ignore
        else:
            # if self.name != "melody":
            #     raise RuntimeError("This model doesn't support melody conditioning. "
            #                        "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        path='null_wav')  # type: ignore
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody.to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device))

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens
    @torch.no_grad()
    def _prepare_tokens_and_attributes_melody_mert1(
            self,
            descriptions: tp.Sequence[tp.Optional[str]],
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
            wav_for_mert=None
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (tp.Optional[torch.Tensor], optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        attributes = [
            ConditioningAttributes_mert1()
            for _ in descriptions]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path='null_wav')  # type: ignore
        else:
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        path='null_wav')  # type: ignore
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody.to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device))
        if wav_for_mert is None:
            for attr in attributes:
                attr.wav_for_mert['wav_for_mert'] = WavCondition(
                    torch.zeros((1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path='null_wav')  # type: ignore
        else:
            assert len(wav_for_mert) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, wav in zip(attributes, wav_for_mert):
                if wav is None:
                    attr.wav_for_mert['wav_for_mert'] = WavCondition(
                        torch.zeros((1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        path='null_wav')  # type: ignore
                else:
                    attr.wav_for_mert['vocal_wav_for_mert1'] = WavCondition(
                        wav.to(device=self.device),
                        torch.tensor([wav.shape[-1]], device=self.device))

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens
    
    @torch.no_grad()
    def _prepare_tokens_and_attributes_audio(
            self,
            audiopaths: tp.Sequence[tp.Optional[str]],
            start_samples,end_samples,
            prompt: tp.Optional[torch.Tensor],
            melody_wavs: tp.Optional[MelodyList] = None,
    ) -> tp.Tuple[tp.List[ConditioningAttributes], tp.Optional[torch.Tensor]]:
        """Prepare model inputs.

        Args:
            descriptions (tp.List[str]): A list of strings used as text conditioning.
            prompt (torch.Tensor): A batch of waveforms used for continuation.
            melody_wavs (tp.Optional[torch.Tensor], optional): A batch of waveforms
                used as melody conditioning. Defaults to None.
        """
        def load_wav_cut(audiopath,start_sample,end_sample,device):
            wav,sr=torchaudio.load(audiopath)
            wav=torch.mean(wav,dim=0)
            wav=wav.squeeze()[start_sample:end_sample].to(device)
            if self.lm.condition_provider.conditioners['wav_for_mert'].resample_rate != sr:
                # print(f'setting rate from {sampling_rate} to {resample_rate}')
                resampler = T.Resample(sr, self.lm.condition_provider.conditioners['wav_for_mert'].resample_rate).to(device)
                wav = resampler(wav)#torch.Size([140520]),5.855s
            else:
                resampler = None
            return wav,torch.tensor([wav.shape[-1]]).to(device),audiopath
        attributes = [
            ConditioningAttributes_mert(wav_for_mert={'wav_for_mert':WavCondition( *load_wav_cut(audiopath,start_sample,end_sample, device=self.device))})
            for audiopath,start_sample,end_sample in zip(audiopaths,start_samples,end_samples)]

        if melody_wavs is None:
            for attr in attributes:
                attr.wav['self_wav'] = WavCondition(
                    torch.zeros((1, 1), device=self.device),
                    torch.tensor([0], device=self.device),
                    path='null_wav')  # type: ignore
        else:
            if self.name != "melody":
                raise RuntimeError("This model doesn't support melody conditioning. "
                                   "Use the `melody` model.")
            assert len(melody_wavs) == len(descriptions), \
                f"number of melody wavs must match number of descriptions! " \
                f"got melody len={len(melody_wavs)}, and descriptions len={len(descriptions)}"
            for attr, melody in zip(attributes, melody_wavs):
                if melody is None:
                    attr.wav['self_wav'] = WavCondition(
                        torch.zeros((1, 1), device=self.device),
                        torch.tensor([0], device=self.device),
                        path='null_wav')  # type: ignore
                else:
                    attr.wav['self_wav'] = WavCondition(
                        melody.to(device=self.device),
                        torch.tensor([melody.shape[-1]], device=self.device))

        if prompt is not None:
            if descriptions is not None:
                assert len(descriptions) == len(prompt), "Prompt and nb. descriptions doesn't match"
            prompt = prompt.to(self.device)
            prompt_tokens, scale = self.compression_model.encode(prompt)
            assert scale is None
        else:
            prompt_tokens = None
        return attributes, prompt_tokens

    def _generate_tokens(self, attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        total_gen_len = int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0

        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback

        if self.duration <= self.max_duration:
            # generate by sampling from LM, simple case.
            with self.autocast:
                gen_tokens = self.lm.generate(
                    prompt_tokens, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)

        else:
            # now this gets a bit messier, we need to handle prompts,
            # melody conditioning etc.
            ref_wavs = [attr.wav['self_wav'] for attr in attributes]
            all_tokens = []
            if prompt_tokens is None:
                prompt_length = 0
            else:
                all_tokens.append(prompt_tokens)
                prompt_length = prompt_tokens.shape[-1]

            stride_tokens = int(self.frame_rate * self.extend_stride)

            while current_gen_offset + prompt_length < total_gen_len:
                time_offset = current_gen_offset / self.frame_rate
                chunk_duration = min(self.duration - time_offset, self.max_duration)
                max_gen_len = int(chunk_duration * self.frame_rate)
                for attr, ref_wav in zip(attributes, ref_wavs):
                    wav_length = ref_wav.length.item()
                    if wav_length == 0:
                        continue
                    # We will extend the wav periodically if it not long enough.
                    # we have to do it here rather than in conditioners.py as otherwise
                    # we wouldn't have the full wav.
                    initial_position = int(time_offset * self.sample_rate)
                    wav_target_length = int(self.max_duration * self.sample_rate)
                    print(initial_position / self.sample_rate, wav_target_length / self.sample_rate)
                    positions = torch.arange(initial_position,
                                             initial_position + wav_target_length, device=self.device)
                    attr.wav['self_wav'] = WavCondition(
                        ref_wav[0][:, positions % wav_length],
                        torch.full_like(ref_wav[1], wav_target_length))
                with self.autocast:
                    gen_tokens = self.lm.generate(
                        prompt_tokens, attributes,
                        callback=callback, max_gen_len=max_gen_len, **self.generation_params)
                if prompt_tokens is None:
                    all_tokens.append(gen_tokens)
                else:
                    all_tokens.append(gen_tokens[:, :, prompt_tokens.shape[-1]:])
                prompt_tokens = gen_tokens[:, :, stride_tokens:]
                prompt_length = prompt_tokens.shape[-1]
                current_gen_offset += stride_tokens

            gen_tokens = torch.cat(all_tokens, dim=-1)

        # generate audio
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio

    def _generate_tokens_v2a_pad6s_5stems(self, codes,attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        
        vocal_max_len=int(self.duration * self.frame_rate)#300
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0
        silent=False,
        max_coarse_history=150  #每次生成有4s的历史信息
        sliding_window_len=150 #每次生成2s
        assert max_coarse_history + sliding_window_len <=vocal_max_len
        max_semantic_history = max_coarse_history
        
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"
        def get_infer_type(idx):
            infertype=['drums','bass','piano','other']
            return infertype[idx]
        def get_pad_type(idx):
            padtype=['vocal','drums','bass','piano']
            return padtype[idx]
        callback = None
        if progress:
            callback = _progress_callback
        vocal_len=codes.shape[-1]#234
        n_steps = vocal_len #要生成的token个数       
        x_semantic_in =codes
        gen_tokens =[ torch.zeros([codes.shape[0],codes.shape[1],0]).to(codes.dtype).to(codes.device) for _ in range(4)]#b,k,0
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0         
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = n_step
            for idx in range(4):#4轨道生成
                x_in = x_semantic_in[:,:, np.max([0, semantic_idx - max_semantic_history]) :]
                x_in = x_in[:, :,:vocal_max_len]
                for jdx in range(idx+1):#获得输入，每个轨道的输入不同，越靠后的输入越长
                    pad_type=get_pad_type(jdx)
                    infer_type=get_infer_type(jdx)
                    pad_token=self.get_pad_token(x_in,pad_type)
                    infer_token=self.get_infer_token(x_in,infer_type)
                    # pad from right side
                    pad_token=torch.cat([pad_token,infer_token],dim=-1)
                    # pad_token = [mid_pad_token[:,:,:1] for _ in range(vocal_max_len-x_in.shape[-1]+4)]#300-234+4
                    # pad_token=torch.cat(pad_token,dim=-1)
                    assert x_in.shape[-1]+pad_token.shape[-1]==(vocal_max_len+4)*(jdx+1)
                    x_in = torch.cat(
                        [
                            x_in,
                            pad_token.cuda(),
                            gen_tokens[jdx][:, :,-max_coarse_history:]
                        ],dim=-1
                    )
                assert x_in.shape[-1]<=(vocal_max_len+4)*(idx+1)+vocal_max_len-sliding_window_len
                total_gen_len=min([vocal_len-n_step,sliding_window_len])#生成的gen_token个数
                with self.autocast:
                    gen_token = self.lm.generate_v2a(
                        x_in, attributes,
                        callback=callback, max_gen_len=total_gen_len, **self.generation_params)
                assert gen_token.shape[-1]==total_gen_len
                gen_tokens[idx] = torch.cat((gen_tokens[idx], gen_token), dim=-1)#更新输出，gen_tokens是4轨道列表
            n_step += sliding_window_len

        # generate audio
        gen_audios=[]
        for g in gen_tokens:
            assert g.dim() == 3
            with torch.no_grad():
                gen_audio = self.compression_model.decode(g, None)
            gen_audios.append(gen_audio)
        # gen_audios = torch.stack(gen_audios)
        # # 计算所有乐器的wav的和
        # gen_audios = gen_audios.sum(dim=0)
        return gen_audios
    def _generate_tokens_v2a_pad6s_va_a(self, codes,attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        
        vocal_max_len=int(self.duration * self.frame_rate)#300
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0
        silent=False,
        max_coarse_history=150  #每次生成有4s的历史信息
        sliding_window_len=150 #每次生成2s
        assert max_coarse_history + sliding_window_len <=vocal_max_len
        max_semantic_history = max_coarse_history
        
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"
        def get_infer_type(idx):
            infertype=['va','a']
            return infertype[idx]
        def get_pad_type(idx):
            padtype=['v','va']
            return padtype[idx]
        callback = None
        if progress:
            callback = _progress_callback
        vocal_len=codes.shape[-1]#234
        n_steps = vocal_len #要生成的token个数       
        x_semantic_in =codes
        gen_tokens =[ torch.zeros([codes.shape[0],codes.shape[1],0]).to(codes.dtype).to(codes.device) for _ in range(2)]#b,k,0
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0     
        def get_pad_token(vocal,audio_type):
            b,K,T=vocal.shape
            if audio_type=='v':
                pad_value=2048
                pad_shape=[b,K,300-T] 
            elif audio_type=='va':
                pad_value=2048
                pad_shape=[b,K,604-T] 
            pad_token=torch.zeros(pad_shape).to(torch.int64)#B,K,K
            pad_token[...]=pad_value
            return pad_token
        def get_infer_token(vocal,audio_type):
            pad_value=2048            
            b,K,T=vocal.shape
            pad_shape=[b,K,K]
            pad_token=torch.zeros(pad_shape).to(torch.int64)#B,K,K
            pad_token[...]=pad_value
            return pad_token    
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = n_step
            for idx in range(2):#4轨道生成
                x_in = x_semantic_in[:,:, np.max([0, semantic_idx - max_semantic_history]) :]
                x_in = x_in[:, :,:vocal_max_len]
                for jdx in range(idx+1):#获得输入，每个轨道的输入不同，越靠后的输入越长
                    pad_type=get_pad_type(jdx)
                    infer_type=get_infer_type(jdx)
                    pad_token=get_pad_token(x_in,pad_type)
                    infer_token=get_infer_token(x_in,infer_type)
                    # pad from right side
                    pad_token=torch.cat([pad_token,infer_token],dim=-1)
                    # pad_token = [mid_pad_token[:,:,:1] for _ in range(vocal_max_len-x_in.shape[-1]+4)]#300-234+4
                    # pad_token=torch.cat(pad_token,dim=-1)
                    assert x_in.shape[-1]+pad_token.shape[-1]==(vocal_max_len+4)*(jdx+1)
                    x_in = torch.cat(
                        [
                            x_in,
                            pad_token.cuda(),
                            gen_tokens[jdx][:, :,-max_coarse_history:]
                        ],dim=-1
                    )
                assert x_in.shape[-1]<=(vocal_max_len+4)*(idx+1)+vocal_max_len-sliding_window_len
                total_gen_len=min([vocal_len-n_step,sliding_window_len])#生成的gen_token个数
                with self.autocast:
                    gen_token = self.lm.generate_v2a(
                        x_in, attributes,
                        callback=callback, max_gen_len=total_gen_len, **self.generation_params)
                assert gen_token.shape[-1]==total_gen_len
                gen_tokens[idx] = torch.cat((gen_tokens[idx], gen_token), dim=-1)#更新输出，gen_tokens是4轨道列表
            n_step += sliding_window_len

        # generate audio
        gen_audios=[]
        for g in gen_tokens:
            assert g.dim() == 3
            with torch.no_grad():
                gen_audio = self.compression_model.decode(g, None)
            gen_audios.append(gen_audio)
        # gen_audios = torch.stack(gen_audios)
        # # 计算所有乐器的wav的和
        # gen_audios = gen_audios.sum(dim=0)
        return gen_audios
    def _generate_tokens_v2a_pad6s(self, codes,mid_pad_token,attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        vocal_max_len=int(self.duration * self.frame_rate)#300
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0
        silent=False,
        max_coarse_history=280  #每次生成有4s的历史信息
        sliding_window_len=20 #每次生成2s
        assert max_coarse_history + sliding_window_len <=vocal_max_len
        max_semantic_history = max_coarse_history
        
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback
        vocal_len=codes.shape[-1]#234
        n_steps = vocal_len #要生成的token个数       
        x_semantic_in =codes
        gen_tokens = torch.zeros([codes.shape[0],codes.shape[1],0]).to(codes.dtype).to(codes.device)#b,k,0
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = n_step
            # pad from right side
            x_in = x_semantic_in[:,:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :,:vocal_max_len]
            pad_token=mid_pad_token[:,:,:1].repeat(1,1,vocal_max_len-x_in.shape[-1]+4)
            # pad_token = [mid_pad_token[:,:,:1] for _ in range(vocal_max_len-x_in.shape[-1]+4)]#300-234+4
            # pad_token=torch.cat(pad_token,dim=-1)
            assert torch.all(mid_pad_token==2048)
            assert x_in.shape[-1]+pad_token.shape[-1]==vocal_max_len+4

            x_in = torch.cat(
                [
                    x_in,
                    pad_token,
                    gen_tokens[:, :,-max_coarse_history:]
                ],dim=-1
            )
            assert x_in.shape[-1]<=2*vocal_max_len+4-sliding_window_len
            total_gen_len=min([vocal_len-n_step,sliding_window_len])#生成的gen_token个数
            with self.autocast:
                gen_token = self.lm.generate_v2a(
                    x_in, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)
                assert gen_token.shape[-1]==total_gen_len
                gen_tokens = torch.cat((gen_tokens, gen_token), dim=-1)
            n_step += sliding_window_len

        # generate audio
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio

    def _generate_tokens_v2a_pad6s_beatnet(self, codes,mid_pad_token,attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        vocal_max_len=int(self.duration * self.frame_rate)#300
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0
        silent=False
        sliding_window_len=20 #每次生成2s
        max_coarse_history=vocal_max_len-sliding_window_len  #每次生成有4s的历史信息
        assert max_coarse_history + sliding_window_len <=vocal_max_len
        max_semantic_history = max_coarse_history
        
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback
        vocal_len=codes.shape[-1]#234
        n_steps = vocal_len #要生成的token个数       
        x_semantic_in =codes
        gen_tokens = torch.zeros([codes.shape[0],codes.shape[1],0]).to(codes.dtype).to(codes.device)#b,k,0
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = n_step
            # pad from right side
            x_in = x_semantic_in[:,:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :,:vocal_max_len]
            pad_token=mid_pad_token[:,:,:1].repeat(1,1,vocal_max_len-x_in.shape[-1]+4)
            # pad_token = [mid_pad_token[:,:,:1] for _ in range(vocal_max_len-x_in.shape[-1]+4)]#300-234+4
            # pad_token=torch.cat(pad_token,dim=-1)
            assert torch.all(mid_pad_token==2048)
            assert x_in.shape[-1]+pad_token.shape[-1]==vocal_max_len+4

            x_in = torch.cat(
                [
                    x_in,
                    pad_token,
                    gen_tokens[:, :,-max_coarse_history:]
                ],dim=-1
            )
            assert x_in.shape[-1]<=2*vocal_max_len+4-sliding_window_len
            total_gen_len=min([vocal_len-n_step,sliding_window_len])#生成的gen_token个数
            with self.autocast:
                gen_token = self.lm.generate_v2a_beatnet(
                    x_in, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)
                assert gen_token.shape[-1]==total_gen_len
                gen_tokens = torch.cat((gen_tokens, gen_token), dim=-1)
            n_step += sliding_window_len

        # generate audio
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio

    def _generate_tokens_v2a(self, codes,mid_pad_token,attributes: tp.List[ConditioningAttributes],
                         prompt_tokens: tp.Optional[torch.Tensor], progress: bool = False) -> torch.Tensor:
        """Generate discrete audio tokens given audio prompt and/or conditions.

        Args:
            attributes (tp.List[ConditioningAttributes]): Conditions used for generation (text/melody).
            prompt_tokens (tp.Optional[torch.Tensor]): Audio prompt used for continuation.
            progress (bool, optional): Flag to display progress of the generation process. Defaults to False.
        Returns:
            torch.Tensor: Generated audio, of shape [B, C, T], T is defined by the generation params.
        """
        vocal_max_len=int(self.duration * self.frame_rate)
        max_prompt_len = int(min(self.duration, self.max_duration) * self.frame_rate)
        current_gen_offset: int = 0
        silent=False,
        max_coarse_history=100  #每次生成有4s的历史信息
        sliding_window_len=200 #每次生成2s
        assert max_coarse_history + sliding_window_len <=vocal_max_len
        max_semantic_history = max_coarse_history
        
        def _progress_callback(generated_tokens: int, tokens_to_generate: int):
            generated_tokens += current_gen_offset
            if self._progress_callback is not None:
                # Note that total_gen_len might be quite wrong depending on the
                # codebook pattern used, but with delay it is almost accurate.
                self._progress_callback(generated_tokens, total_gen_len)
            else:
                print(f'{generated_tokens: 6d} / {total_gen_len: 6d}', end='\r')

        if prompt_tokens is not None:
            assert max_prompt_len >= prompt_tokens.shape[-1], \
                "Prompt is longer than audio to generate"

        callback = None
        if progress:
            callback = _progress_callback
        vocal_len=codes.shape[-1]
        n_steps = vocal_len #要生成的token个数       
        x_semantic_in =codes
        gen_tokens = torch.zeros([codes.shape[0],codes.shape[1],0]).to(codes.dtype).to(codes.device)#b,k,0
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for _ in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = n_step
            # pad from right side
            x_in = x_semantic_in[:,:, np.max([0, semantic_idx - max_semantic_history]) :]
            x_in = x_in[:, :,:vocal_max_len]
            x_in = torch.cat(
                [
                    x_in,
                    mid_pad_token,
                    gen_tokens[:, :,-max_coarse_history:]
                ],dim=-1
            )
            total_gen_len=min([vocal_len-n_step,sliding_window_len])
            with self.autocast:
                gen_token = self.lm.generate_v2a(
                    x_in, attributes,
                    callback=callback, max_gen_len=total_gen_len, **self.generation_params)
            
                gen_tokens = torch.cat((gen_tokens, gen_token), dim=-1)
            n_step += sliding_window_len

        # generate audio
        assert gen_tokens.dim() == 3
        with torch.no_grad():
            gen_audio = self.compression_model.decode(gen_tokens, None)
        return gen_audio
