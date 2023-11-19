# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from functools import partial
import logging
import math
import typing as tp

import torch
from torch import nn
import torchaudio
from ..utils import utils
from ..modules.streaming import StreamingModule, State
from ..modules.transformer import StreamingTransformer, create_norm_fn
from ..modules.conditioners import (
    ConditionFuser,
    ClassifierFreeGuidanceDropout,
    AttributeDropout,
    ConditioningProvider,
    ConditioningAttributes,
    ConditionType,
)
from ..modules.codebooks_patterns import CodebooksPatternProvider
from ..modules.activations import get_activation_fn

from BeatNet.BeatNet import BeatNet
from transformers import Wav2Vec2FeatureExtractor
import torchaudio.transforms as T
import sys
sys.path.append('/data/huangrm/audioLM/bark_trainer_hubert/bark-gui/models')
from MERT_v0_public.cqt import  CQT
from MERT_v0_public.configuration_MERT import  MERTConfig

logger = logging.getLogger(__name__)
ConditionTensors = tp.Dict[str, ConditionType]
CFGConditions = tp.Union[ConditionTensors, tp.Tuple[ConditionTensors, ConditionTensors]]


def get_init_fn(method: str, input_dim: int, init_depth: tp.Optional[int] = None):
    """LM layer initialization.
    Inspired from xlformers: https://github.com/fairinternal/xlformers

    Args:
        method (str): Method name for init function. Valid options are:
            'gaussian', 'uniform'.
        input_dim (int): Input dimension of the initialized module.
        init_depth (Optional[int]): Optional init depth value used to rescale
            the standard deviation if defined.
    """
    # Compute std
    std = 1 / math.sqrt(input_dim)
    # Rescale with depth
    if init_depth is not None:
        std = std / math.sqrt(2 * init_depth)

    if method == 'gaussian':
        return partial(
            torch.nn.init.trunc_normal_, mean=0.0, std=std, a=-3 * std, b=3 * std
        )
    elif method == 'uniform':
        bound = math.sqrt(3) * std  # ensure the standard deviation is `std`
        return partial(torch.nn.init.uniform_, a=-bound, b=bound)
    else:
        raise ValueError("Unsupported layer initialization method")


def init_layer(m: nn.Module,
               method: str,
               init_depth: tp.Optional[int] = None,
               zero_bias_init: bool = False):
    """Wrapper around ``get_init_fn`` for proper initialization of LM modules.

    Args:
        m (nn.Module): Module to initialize.
        method (str): Method name for the init function.
        init_depth (Optional[int]): Optional init depth value used to rescale
            the standard deviation if defined.
        zero_bias_init (bool): Whether to initialize the bias to 0 or not.
    """
    if isinstance(m, nn.Linear):
        init_fn = get_init_fn(method, m.in_features, init_depth=init_depth)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)
        if zero_bias_init and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        init_fn = get_init_fn(method, m.embedding_dim, init_depth=None)
        if m.weight.device.type == 'cpu' and m.weight.dtype == torch.float16:
            weight = m.weight.float()
            init_fn(weight)
            m.weight.data[:] = weight.half()
        else:
            init_fn(m.weight)

class Beatnet():
    def __init__(self,frame_rate=50 ):
        # super().__init__()
        self.estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        self.frame_rate=frame_rate
    def get_beat_frame(self,audio_path,max_len):
        def beat_to_frame(beat,max_len):
            beat_frame=torch.zeros([max_len],dtype=torch.int64)
            for b in beat:
                assert len(b)==2
                t=b[0]
                beat_type=b[1]
                beat_frame[int(t*self.frame_rate)]=int(beat_type)
            return beat_frame
        # audio_path='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test/光年-叶琼琳_00:58.74_01:02.25/accompaniment.wav'
        beat = self.estimator.process(audio_path)#shape:(8, 2)
        beat_frame=beat_to_frame(beat,max_len)
        return beat_frame
    def get_batch_beat_frame(self,audio_paths,max_len,n_books):
        duration=int((max_len-n_books)/2/50)
        assert duration==6
        def batch_beat_to_frame(beat,max_len):
            beat_frame=torch.zeros([max_len],dtype=torch.int64)
            for b in beat:
                assert len(b)==2
                t=b[0]
                beat_type=b[1]
                if n_books+duration*50+1+int(t*self.frame_rate)<max_len:
                    beat_frame[n_books+duration*50+1+int(t*self.frame_rate)]=int(beat_type)
            return beat_frame
        beat_frames=[]
        for audio_path in audio_paths:
            beat = self.estimator.process(audio_path)#shape:(8, 2)
            beat_frames.append(batch_beat_to_frame(beat,max_len))
        beat_frames=torch.stack(beat_frames,dim=0)
        return beat_frames
class ScaledEmbedding(nn.Embedding):
    """Boost learning rate for embeddings (with `scale`).
    """
    def __init__(self, *args, lr=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr = lr

    def make_optim_group(self):
        group = {"params": list(self.parameters())}
        if self.lr is not None:
            group["lr"] = self.lr
        return group


@dataclass
class LMOutput:
    # The logits are already re-aligned with the input codes
    # hence no extra shift is required, e.g. when computing CE
    logits: torch.Tensor  # [B, K, T, card]
    mask: torch.Tensor  # [B, K, T]


class LMModel(StreamingModule):
    """Transformer-based language model on multiple streams of codes.

    Args:
        pattern_provider (CodebooksPatternProvider): Pattern provider for codebook interleaving.
        condition_provider (MusicConditioningProvider): Conditioning provider from metadata.
        fuser (ConditionFuser): Fuser handling the fusing of conditions with language model input.
        n_q (int): Number of parallel streams to model.
        card (int): Cardinality, vocabulary size.
        dim (int): Dimension of the transformer encoder.
        num_heads (int): Number of heads for the transformer encoder.
        hidden_scale (int): Scale for hidden feed forward dimension of the transformer encoder.
        norm (str): Normalization method.
        norm_first (bool): Use pre-norm instead of post-norm.
        emb_lr (Optional[float]): Embedding-specific learning rate.
        bias_proj (bool): Use bias for output projections.
        weight_init (Optional[str]): Method for weight initialization.
        depthwise_init (Optional[str]): Method for depthwise weight initialization.
        zero_bias_init (bool): If true and bias in Linears, initialize bias to zeros.
        cfg_dropout (float): Classifier-free guidance dropout.
        cfg_coef (float): Classifier-free guidance coefficient.
        attribute_dropout (dict): Attribute dropout probabilities.
        two_step_cfg (bool): Whether to run classifier free-guidance with 2 distinct steps.
        **kwargs: Additional parameters for the transformer encoder.
    """
    def __init__(self, pattern_provider: CodebooksPatternProvider, condition_provider: ConditioningProvider,
                 fuser: ConditionFuser, n_q: int = 8, card: int = 1024, dim: int = 128, num_heads: int = 8,
                 hidden_scale: int = 4, norm: str = 'layer_norm', norm_first: bool = False,
                 emb_lr: tp.Optional[float] = None, bias_proj: bool = True,
                 weight_init: tp.Optional[str] = None, depthwise_init: tp.Optional[str] = None,
                 zero_bias_init: bool = False, cfg_dropout: float = 0, cfg_coef: float = 1.0,
                 attribute_dropout: tp.Dict[str, tp.Dict[str, float]] = {}, two_step_cfg: bool = False,stems=2,
                 mode=None,           
                 **kwargs):
        super().__init__()
        self.cfg_coef = cfg_coef
        self.cfg_dropout = ClassifierFreeGuidanceDropout(p=cfg_dropout)
        self.att_dropout = AttributeDropout(p=attribute_dropout)
        self.condition_provider = condition_provider
        self.fuser = fuser
        self.card = card
        if stems==5:
            embed_dim = self.card + 8
        else:
            embed_dim = self.card + 1
        self.n_q = n_q
        self.dim = dim
        self.pattern_provider = pattern_provider
        self.two_step_cfg = two_step_cfg
        self.emb = nn.ModuleList([ScaledEmbedding(embed_dim, dim, lr=emb_lr) for _ in range(n_q)])#embed_dim2049 ->+4+4-1=+7=4pad+3infer
        if 'activation' in kwargs:
            kwargs['activation'] = get_activation_fn(kwargs['activation'])
        self.transformer = StreamingTransformer(
            d_model=dim, num_heads=num_heads, dim_feedforward=int(hidden_scale * dim),
            norm=norm, norm_first=norm_first, **kwargs)
        self.out_norm: tp.Optional[nn.Module] = None
        if norm_first:
            self.out_norm = create_norm_fn(norm, dim)
        if mode=='beatnet':
            self.linear_beatnet =nn.Linear(dim, 5, bias=True)
            self.emb_beatnet=nn.Embedding(5, dim)
            self.beatnet=Beatnet()
        elif mode=='beatnet_positionemb':
            self.linear_beatnet =nn.Linear(dim, 5, bias=True)
            self.emb_beatnet=nn.Embedding(5, dim)
            self.beatnet=Beatnet()
            
            self.position_emb=nn.Embedding(config.block_size, config.n_embd)
        elif mode=='beatnet_cqtmert':
            self.linear_beatnet =nn.Linear(dim, 5, bias=True)
            self.emb_beatnet=nn.Embedding(5, dim)
            self.beatnet=Beatnet()
            
            model_path='/data/huangrm/audioLM/bark_trainer_hubert/bark-gui/models/MERT_v0_public'
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path,trust_remote_code=True)
            self.resample_rate = self.processor.sampling_rate#16000
            self.sampling_rate =44100
            self.resampler = T.Resample(self.sampling_rate, self.resample_rate)
            self.cqtconfig=MERTConfig.from_pretrained(model_path)
            assert self.cqtconfig.sample_rate==self.resample_rate
            self.cqt=CQT(self.cqtconfig)
            self.linear_cqt=nn.Linear(dim, 336, bias=True)
            
        self.linears = nn.ModuleList([nn.Linear(dim, self.card, bias=bias_proj) for _ in range(n_q)])
        self._init_weights(weight_init, depthwise_init, zero_bias_init)
        self._fsdp: tp.Optional[nn.Module]
        self.__dict__['_fsdp'] = None
        self.mode=mode
            
        

    def _init_weights(self, weight_init: tp.Optional[str], depthwise_init: tp.Optional[str], zero_bias_init: bool):
        """Initialization of the transformer module weights.

        Args:
            weight_init (Optional[str]): Weight initialization strategy. See ``get_init_fn`` for valid options.
            depthwise_init (Optional[str]): Depwthwise initialization strategy. The following options are valid:
                'current' where the depth corresponds to the current layer index or 'global' where the total number
                of layer is used as depth. If not set, no depthwise initialization strategy is used.
            zero_bias_init (bool): Whether to initalize bias to zero or not.
        """
        assert depthwise_init is None or depthwise_init in ['current', 'global']
        assert depthwise_init is None or weight_init is not None, \
            "If 'depthwise_init' is defined, a 'weight_init' method should be provided."
        assert not zero_bias_init or weight_init is not None, \
            "If 'zero_bias_init', a 'weight_init' method should be provided"

        if weight_init is None:
            return

        for emb_layer in self.emb:
            init_layer(emb_layer, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

        for layer_idx, tr_layer in enumerate(self.transformer.layers):
            depth = None
            if depthwise_init == 'current':
                depth = layer_idx + 1
            elif depthwise_init == 'global':
                depth = len(self.transformer.layers)
            init_fn = partial(init_layer, method=weight_init, init_depth=depth, zero_bias_init=zero_bias_init)
            tr_layer.apply(init_fn)

        for linear in self.linears:
            init_layer(linear, method=weight_init, init_depth=None, zero_bias_init=zero_bias_init)

    @property
    def special_token_id(self) -> int:
        return self.card

    @property
    def num_codebooks(self) -> int:
        return self.n_q

    def forward(self, sequence: torch.Tensor,
                conditions: tp.List[ConditioningAttributes],
                condition_tensors: tp.Optional[ConditionTensors] = None,beat=None,cqt=None) -> torch.Tensor:
        """Apply language model on sequence and conditions.
        Given a tensor of sequence of shape [B, K, S] with K the number of codebooks and
        S the sequence steps, return the logits with shape [B, card, K, S].

        Args:
            indices (torch.Tensor): indices of the codes to model.
            conditions (list[ConditioningAttributes]): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType] or None): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            torch.Tensor: Logits.
        """
        B, K, S = sequence.shape
        assert K == self.num_codebooks, 'Sequence shape must match the specified number of codebooks'
        # if self.mode=='beatnet':
        #     assert beat !=None
        #     input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)]+[self.emb_beatnet(beat)])#([B, 1, 1536])
        # else:
        input_ = sum([self.emb[k](sequence[:, k]) for k in range(K)])#([B, 1, 1536])
        if condition_tensors is None:
            assert not self._is_streaming, "Conditions tensors should be precomputed when streaming."
            # apply dropout modules
            conditions = self.cfg_dropout(conditions)
            conditions = self.att_dropout(conditions)
            tokenized = self.condition_provider.tokenize(conditions)
            # encode conditions and fuse, both have a streaming cache to not recompute when generating.
            condition_tensors = self.condition_provider(tokenized)
        else:
            assert not conditions, "Shouldn't pass both conditions and condition_tensors."

        input_, cross_attention_input = self.fuser(input_, condition_tensors)#cat(chroma,desc,input_) 235+1+604

        out = self.transformer(input_, cross_attention_src=cross_attention_input)
        if self.out_norm:
            out = self.out_norm(out)
        logits = torch.stack([self.linears[k](out) for k in range(K)], dim=1)  # [B, K, S, card]
        if self.mode=='beatnet':
            beat_logits=self.linear_beatnet(out)
        elif self.mode=='beatnet_cqtmert':
            beat_logits=self.linear_beatnet(out)
            cqt_pred=self.linear_cqt(out)
        elif self.mode=='beatnet_positionemb':
            beat_logits=self.linear_beatnet(out)
            self.position_emb=nn.Embedding(config.block_size, config.n_embd),
            assert 1==2
        # remove the prefix from the model outputs
        if len(self.fuser.fuse2cond['prepend']) > 0:
            logits = logits[:, :, -S:]
        if  self.mode=='beatnet':
            return logits,beat_logits
        elif self.mode=='beatnet_cqtmert':
            return logits,beat_logits,cqt_pred
        return logits  # [B, K, S, card]
    def get_batch_cqt(self,audio_paths,max_len,n_books):
        duration=int((max_len-n_books)/2/50)
        assert duration==15
        def pad_cqt(cqt,max_len):
            cqt_frame=torch.zeros([max_len,336],dtype=cqt.dtype,device=cqt.device)#50hz
            start_id=n_books+duration*50+1
            cqt=cqt.squeeze()
            if cqt.shape[0]>=duration*50:#trim or pad到750长度
                cqt=cqt[:duration*50,:]
            else:
                cqt=torch.nn.functional.pad(cqt,(0,0,0,duration*50-cqt.shape[0]),value=0)
            cqt_frame[start_id:,:]=cqt
            return cqt_frame
        cqts=[]
        for audio_path in audio_paths:
            wav,sr=torchaudio.load(audio_path)
            wav=wav.cuda()
            if len(wav.shape)==2:#转化为单声道
                wav=torch.mean(wav,dim=0)
            input_audio = self.resampler(wav)#torch.Size([140520]),5.855s
            assert sr==self.resampler.orig_freq
            inputs = self.processor(input_audio, sampling_rate=self.resample_rate, return_tensors="pt")#torch.Size([140520]),5.855s
            inputs['input_values']=inputs['input_values'].cuda()
            inputs['attention_mask']=inputs['attention_mask'].cuda()
            with torch.no_grad():
                cqt = self.cqt(**inputs, output_hidden_states=True)
            cqts.append(pad_cqt(cqt,max_len))
        cqts=torch.stack(cqts,dim=0)
        return cqts
    def compute_predictions_beatnet_cqtmert(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None,audiopaths=None) -> LMOutput:

        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list[ConditioningAttributes]): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType] or None): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        
        B, K, T = codes.shape
        beat=self.beatnet.get_batch_beat_frame(audiopaths,T+1,K)
        beat=beat.cuda()
        cqt=self.get_batch_cqt(audiopaths,T+1,K)
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits,beat_logits,cqt_pred = model(sequence_codes, conditions, condition_tensors,beat=beat,cqt=cqt)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask),beat_logits[:,:-1,:],beat[:,1:],cqt_pred[:,:-1,:],cqt[:,1:,:]

    
    def compute_predictions_beatnet(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None,audiopaths=None) -> LMOutput:

        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list[ConditioningAttributes]): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType] or None): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        
        B, K, T = codes.shape
        beat=self.beatnet.get_batch_beat_frame(audiopaths,T+1,K)
        beat=beat.cuda()
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits,beat_logits = model(sequence_codes, conditions, condition_tensors,beat=beat)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask),beat_logits[:,:-1,:],beat[:,1:]

    def compute_predictions(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list[ConditioningAttributes]): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType] or None): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask)
    def compute_predictions_mert(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list[ConditioningAttributes]): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType] or None): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes, conditions, condition_tensors)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask)

    def compute_predictions_v2a(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list[ConditioningAttributes]): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType] or None): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        device=codes.device
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes[:,:,1:], conditions, condition_tensors)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        logits=torch.cat([logits,torch.zeros([B,2048,4,1]).to(device)],dim=-1)
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        return LMOutput(logits, logits_mask)

    def _sample_next_token(self,
                           sequence: torch.Tensor,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (Dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        if self.two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting both conditional and unconditional logits.
                sequence = torch.cat([sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token
    
    def _sample_next_token_beatnet(self,
                           sequence: torch.Tensor,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (Dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        if self.two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting both conditional and unconditional logits.
                sequence = torch.cat([sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, uncond_logits = all_logits[0].split(B, dim=0)  # [B, K, T, card]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        logits = logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token
    
    def _sample_token(self,
                           sequence: torch.Tensor,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (Dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        if self.two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting both conditional and unconditional logits.
                sequence = torch.cat([sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        # logits = logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token

    @torch.no_grad()
    def generate(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 two_step_cfg: bool = False,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (Optional[torch.Tensor]): Prompt tokens of shape [B, K, T].
            conditions_tensors (Dict[str, torch.Tensor]): Set of conditions or None.
            num_samples (int or None): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            remove_prompts (bool): Whether to remove prompts from generation or not.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistents.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsitent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        # We also support doing two different passes, in particular to ensure that
        # the padding structure is exactly the same between train anf test.
        # With a batch size of 1, this can be slower though.
        cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step_cfg:
                cfg_conditions = (
                    self.condition_provider(self.condition_provider.tokenize(conditions)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                )
            else:
                conditions = conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions)
                cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        assert start_offset < max_gen_len

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        with self.streaming():
            unconditional_state = self.get_streaming_state()
            prev_offset = 0
            gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
            for offset in range(start_offset_sequence, gen_sequence_len):
                # get current sequence (note that the streaming API is providing the caching over previous offsets)
                curr_sequence = gen_sequence[..., prev_offset:offset]
                curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
                if check:
                    # check coherence between mask and sequence
                    assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                    # should never happen as gen_sequence is filled progressively
                    assert not (curr_sequence == unknown_token).any()
                # sample next token from the model, next token shape is [B, K, 1]
                next_token = self._sample_next_token(
                    curr_sequence, cfg_conditions, unconditional_state, use_sampling, temp, top_k, top_p,
                    cfg_coef=cfg_coef)
                # ensure the tokens that should be masked are properly set to special_token_id
                # as the model never output special_token_id
                valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
                next_token[~valid_mask] = self.special_token_id
                # ensure we don't overwrite prompt tokens, we only write over unknown tokens
                # (then mask tokens should be left as is as well, which is correct)
                gen_sequence[..., offset:offset+1] = torch.where(
                    gen_sequence[..., offset:offset+1] == unknown_token,
                    next_token, gen_sequence[..., offset:offset+1]
                )
                prev_offset = offset
                if callback is not None:
                    callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()

        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes
    
    @torch.no_grad()
    def generate_v2a(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 two_step_cfg: bool = False,
                 remove_prompts: bool = True,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (Optional[torch.Tensor]): Prompt tokens of shape [B, K, T].
            conditions_tensors (Dict[str, torch.Tensor]): Set of conditions or None.
            num_samples (int or None): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            remove_prompts (bool): Whether to remove prompts from generation or not.
        Returns:
            torch.Tensor: Generated tokens.
        """
        
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistents.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsitent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        # We also support doing two different passes, in particular to ensure that
        # the padding structure is exactly the same between train anf test.
        # With a batch size of 1, this can be slower though.
        cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step_cfg:
                cfg_conditions = (
                    self.condition_provider(self.condition_provider.tokenize(conditions)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                )
            else:
                conditions = conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions)
                cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        # assert start_offset < max_gen_len
        max_gen_len=start_offset+max_gen_len
        # max_gen_len=start_offset*2-K
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        # with self.streaming():
        unconditional_state = self.get_streaming_state()
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
        for offset in range(start_offset_sequence, gen_sequence_len):#要预测id=start_offset_sequence-1处的token
            # get current sequence (note that the streaming API is providing the caching over previous offsets)
            curr_sequence = gen_sequence[..., prev_offset:offset]
            curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
            if check:
                # check coherence between mask and sequence
                assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                # should never happen as gen_sequence is filled progressively
                assert not (curr_sequence == unknown_token).any()
            # sample next token from the model, next token shape is [B, K, 1]
            next_token = self._sample_next_token(
                curr_sequence, cfg_conditions, unconditional_state, use_sampling, temp, top_k, top_p,
                cfg_coef=cfg_coef)
            # ensure the tokens that should be masked are properly set to special_token_id
            # as the model never output special_token_id
            valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
            next_token[~valid_mask] = self.special_token_id
            # ensure we don't overwrite prompt tokens, we only write over unknown tokens
            # (then mask tokens should be left as is as well, which is correct)
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] != 2048,
                next_token, gen_sequence[..., offset:offset+1]
            )
            # gen_sequence[..., offset:offset+1] = next_token
            # prev_offset = offset
            if callback is not None:
                callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()

        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes
    @torch.no_grad()
    def generate_v2a_beatnet(self,
                 prompt: tp.Optional[torch.Tensor] = None,
                 conditions: tp.List[ConditioningAttributes] = [],
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,
                 use_sampling: bool = True,
                 temp: float = 1.0,
                 top_k: int = 250,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 two_step_cfg: bool = False,
                 remove_prompts: bool = True,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (Optional[torch.Tensor]): Prompt tokens of shape [B, K, T].
            conditions_tensors (Dict[str, torch.Tensor]): Set of conditions or None.
            num_samples (int or None): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            remove_prompts (bool): Whether to remove prompts from generation or not.
        Returns:
            torch.Tensor: Generated tokens.
        """
        
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistents.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))
        else:
            possible_num_samples.append(1)
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsitent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        # We also support doing two different passes, in particular to ensure that
        # the padding structure is exactly the same between train anf test.
        # With a batch size of 1, this can be slower though.
        cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step_cfg:
                cfg_conditions = (
                    self.condition_provider(self.condition_provider.tokenize(conditions)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                )
            else:
                conditions = conditions + null_conditions
                tokenized = self.condition_provider.tokenize(conditions)
                cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape
        start_offset = T
        # assert start_offset < max_gen_len
        max_gen_len=start_offset+max_gen_len
        # max_gen_len=start_offset*2-K
        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)
        assert start_offset_sequence is not None

        # with self.streaming():
        unconditional_state = self.get_streaming_state()
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
        for offset in range(start_offset_sequence, gen_sequence_len):#要预测id=start_offset_sequence-1处的token
            # get current sequence (note that the streaming API is providing the caching over previous offsets)
            curr_sequence = gen_sequence[..., prev_offset:offset]
            curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
            if check:
                # check coherence between mask and sequence
                assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                # should never happen as gen_sequence is filled progressively
                assert not (curr_sequence == unknown_token).any()
            # sample next token from the model, next token shape is [B, K, 1]
            next_token = self._sample_next_token_beatnet(
                curr_sequence, cfg_conditions, unconditional_state, use_sampling, temp, top_k, top_p,
                cfg_coef=cfg_coef)
            # ensure the tokens that should be masked are properly set to special_token_id
            # as the model never output special_token_id
            valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)
            next_token[~valid_mask] = self.special_token_id
            # ensure we don't overwrite prompt tokens, we only write over unknown tokens
            # (then mask tokens should be left as is as well, which is correct)
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] != 2048,
                next_token, gen_sequence[..., offset:offset+1]
            )
            # gen_sequence[..., offset:offset+1] = next_token
            # prev_offset = offset
            if callback is not None:
                callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()

        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)

        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes
     
    @torch.no_grad()
    def generate_no_stream(self,
                 prompt: tp.Optional[torch.Tensor] = None,#None,prime wav
                 conditions: tp.List[ConditioningAttributes] = [],#包含text
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,#设置的生产长度6s*frame_rate:6*50
                 use_sampling: bool = True,#采样
                 temp: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 two_step_cfg: bool = False,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 cfg_conditions=None) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (Optional[torch.Tensor]): Prompt tokens of shape [B, K, T]. # prime_wav
            conditions_tensors (Dict[str, torch.Tensor]): Set of conditions or None.
            num_samples (int or None): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            remove_prompts (bool): Whether to remove prompts from generation or not.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistents.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))#样本数量
        else:
            possible_num_samples.append(1)#无条件生成一个wav
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsitent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        # We also support doing two different passes, in particular to ensure that
        # the padding structure is exactly the same between train anf test.
        # With a batch size of 1, this can be slower though.
        cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg  # false
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step_cfg:
                cfg_conditions = (
                    self.condition_provider(self.condition_provider.tokenize(conditions)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                )
            else:
                if cfg_conditions==None:
                    conditions = conditions 
                    tokenized = self.condition_provider.tokenize(conditions)
                    cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape#K 8codeboo数;T序列长度
        start_offset = T#0
        assert start_offset < max_gen_len

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)#B,4,300
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        # gen_sequence:#B,4,304     
        # tensor([[2048,   -1,   -1,  ..., 2048, 2048, 2048],
        #         [2048, 2048,   -1,  ...,   -1, 2048, 2048],
        #         [2048, 2048, 2048,  ...,   -1,   -1, 2048],
        #         [2048, 2048, 2048,  ...,   -1,   -1,   -1]], device='cuda:0')
        
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)#1
        assert start_offset_sequence is not None

        unconditional_state = self.get_streaming_state()
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
        for offset in range(start_offset_sequence, gen_sequence_len):#offset range(1,304)
            # get current sequence (note that the streaming API is providing the caching over previous offsets)
            curr_sequence = gen_sequence[..., prev_offset:offset]
            curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
            if check:
                # check coherence between mask and sequence
                assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                # should never happen as gen_sequence is filled progressively
                assert not (curr_sequence == unknown_token).any()
            # sample next token from the model, next token shape is [B, K, 1]
            next_token = self._sample_next_token(
                curr_sequence, cfg_conditions, unconditional_state, use_sampling, temp, top_k, top_p,
                cfg_coef=cfg_coef)#B,4,1
            # ensure the tokens that should be masked are properly set to special_token_id
            # as the model never output special_token_id
            valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)#当前生成的token里有效的
            next_token[~valid_mask] = self.special_token_id#无效位置生成的token变为2048
            # ensure we don't overwrite prompt tokens, we only write over unknown tokens
            # (then mask tokens should be left as is as well, which is correct)
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] == unknown_token,
                next_token, gen_sequence[..., offset:offset+1]
            )#更新
            # prev_offset = offset
            if callback is not None:
                callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
        unconditional_state.clear()

        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)#去掉2048，对齐时间戳
        #out_codes: B,4,3000.去除2048，对齐gen_sequence
        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes
    @torch.no_grad()
    def generate_v2a0(self,
                     vocal_path,
                 prompt: tp.Optional[torch.Tensor] = None,#None,prime wav
                 conditions: tp.List[ConditioningAttributes] = [],#包含text
                 num_samples: tp.Optional[int] = None,
                 max_gen_len: int = 256,#设置的生产长度6s*frame_rate:6*50
                 use_sampling: bool = True,#采样
                 temp: float = 1.0,
                 top_k: int = 0,
                 top_p: float = 0.0,
                 cfg_coef: tp.Optional[float] = None,
                 two_step_cfg: bool = False,
                 remove_prompts: bool = False,
                 check: bool = False,
                 callback: tp.Optional[tp.Callable[[int, int], None]] = None,
                 cfg_conditions=None) -> torch.Tensor:
        """Generate tokens sampling from the model given a prompt or unconditionally. Generation can
        be perform in a greedy fashion or using sampling with top K and top P strategies.

        Args:
            prompt (Optional[torch.Tensor]): Prompt tokens of shape [B, K, T]. # prime_wav
            conditions_tensors (Dict[str, torch.Tensor]): Set of conditions or None.
            num_samples (int or None): Number of samples to generate when no prompt and no conditions are given.
            max_gen_len (int): Maximum generation length.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            remove_prompts (bool): Whether to remove prompts from generation or not.
        Returns:
            torch.Tensor: Generated tokens.
        """
        assert not self.training, "generation shouldn't be used in training mode."
        first_param = next(iter(self.parameters()))
        device = first_param.device

        # Checking all input shapes are consistents.
        possible_num_samples = []
        if num_samples is not None:
            possible_num_samples.append(num_samples)
        elif prompt is not None:
            possible_num_samples.append(prompt.shape[0])
        elif conditions:
            possible_num_samples.append(len(conditions))#样本数量
        else:
            possible_num_samples.append(1)#无条件生成一个wav
        assert [x == possible_num_samples[0] for x in possible_num_samples], "Inconsitent inputs shapes"
        num_samples = possible_num_samples[0]

        # below we create set of conditions: one conditional and one unconditional
        # to do that we merge the regular condition together with the null condition
        # we then do 1 forward pass instead of 2.
        # the reason for that is two-fold:
        # 1. it is about x2 faster than doing 2 forward passes
        # 2. avoid the streaming API treating the 2 passes as part of different time steps
        # We also support doing two different passes, in particular to ensure that
        # the padding structure is exactly the same between train anf test.
        # With a batch size of 1, this can be slower though.
        cfg_conditions: CFGConditions
        two_step_cfg = self.two_step_cfg if two_step_cfg is None else two_step_cfg  # false
        if conditions:
            null_conditions = ClassifierFreeGuidanceDropout(p=1.0)(conditions)
            if two_step_cfg:
                cfg_conditions = (
                    self.condition_provider(self.condition_provider.tokenize(conditions)),
                    self.condition_provider(self.condition_provider.tokenize(null_conditions)),
                )
            else:
                if cfg_conditions==None:
                    conditions = conditions 
                    tokenized = self.condition_provider.tokenize(conditions)
                    cfg_conditions = self.condition_provider(tokenized)
        else:
            cfg_conditions = {}

        if prompt is None:
            assert num_samples > 0
            prompt = torch.zeros((num_samples, self.num_codebooks, 0), dtype=torch.long, device=device)

        B, K, T = prompt.shape#K 8codeboo数;T序列长度
        start_offset = T#0
        assert start_offset < max_gen_len

        pattern = self.pattern_provider.get_pattern(max_gen_len)
        # this token is used as default value for codes that are not generated yet
        unknown_token = -1

        # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
        gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)#B,4,300
        # filling the gen_codes with the prompt if needed
        gen_codes[..., :start_offset] = prompt
        # create the gen_sequence with proper interleaving from the pattern: [B, K, S]
        gen_sequence, indexes, mask = pattern.build_pattern_sequence(gen_codes, self.special_token_id)
        # gen_sequence:#B,4,304     
        # tensor([[2048,   -1,   -1,  ..., 2048, 2048, 2048],
        #         [2048, 2048,   -1,  ...,   -1, 2048, 2048],
        #         [2048, 2048, 2048,  ...,   -1,   -1, 2048],
        #         [2048, 2048, 2048,  ...,   -1,   -1,   -1]], device='cuda:0')
        
        # retrieve the start_offset in the sequence:
        # it is the first sequence step that contains the `start_offset` timestep
        start_offset_sequence = pattern.get_first_step_with_timesteps(start_offset)#1
        assert start_offset_sequence is not None

        unconditional_state = self.get_streaming_state()
        prev_offset = 0
        gen_sequence_len = gen_sequence.shape[-1]  # gen_sequence shape is [B, K, S]
        
        
        start_offset_sequence=gen_sequence_len-1
        for offset in range(start_offset_sequence, gen_sequence_len):#offset range(1,304)
            # get current sequence (note that the streaming API is providing the caching over previous offsets)
            curr_sequence = gen_sequence[..., prev_offset:offset]
            curr_mask = mask[None, ..., prev_offset:offset].expand(B, -1, -1)
            if check:
                # check coherence between mask and sequence
                assert (curr_sequence == torch.where(curr_mask, curr_sequence, self.special_token_id)).all()
                # should never happen as gen_sequence is filled progressively
                assert not (curr_sequence == unknown_token).any()
            # sample next token from the model, next token shape is [B, K, 1]
            next_token = self._sample_token(
                curr_sequence, cfg_conditions, unconditional_state, use_sampling, temp, top_k, top_p,
                cfg_coef=cfg_coef)#B,4,1
            # ensure the tokens that should be masked are properly set to special_token_id
            # as the model never output special_token_id
            valid_mask = mask[..., offset:offset+1].expand(B, -1, -1)#当前生成的token里有效的
            next_token[~valid_mask] = self.special_token_id#无效位置生成的token变为2048
            # ensure we don't overwrite prompt tokens, we only write over unknown tokens
            # (then mask tokens should be left as is as well, which is correct)
            gen_sequence[..., offset:offset+1] = torch.where(
                gen_sequence[..., offset:offset+1] == unknown_token,
                next_token, gen_sequence[..., offset:offset+1]
            )#更新
            # prev_offset = offset
            if callback is not None:
                callback(1 + offset - start_offset_sequence, gen_sequence_len - start_offset_sequence)
                
                
                
                
        unconditional_state.clear()

        # ensure sequence has been entirely filled
        assert not (gen_sequence == unknown_token).any()
        # ensure gen_sequence pattern and mask are matching
        # which means the gen_sequence is valid according to the pattern
        assert (
            gen_sequence == torch.where(mask[None, ...].expand(B, -1, -1), gen_sequence, self.special_token_id)
        ).all()
        # get back the codes, trimming the prompt if needed and cutting potentially incomplete timesteps
        out_codes, out_indexes, out_mask = pattern.revert_pattern_sequence(gen_sequence, special_token=unknown_token)#去掉2048，对齐时间戳
        #out_codes: B,4,3000.去除2048，对齐gen_sequence
        # sanity checks over the returned codes and corresponding masks
        assert (out_codes[..., :max_gen_len] != unknown_token).all()
        assert (out_mask[..., :max_gen_len] == 1).all()

        out_start_offset = start_offset if remove_prompts else 0
        out_codes = out_codes[..., out_start_offset:max_gen_len]

        # ensure the returned codes are all valid
        assert (out_codes >= 0).all() and (out_codes <= self.card).all()
        return out_codes

    def _sample_token_v2a(self,
                           sequence: torch.Tensor,
                           cfg_conditions: CFGConditions,
                           unconditional_state: State,
                           use_sampling: bool = False,
                           temp: float = 1.0,
                           top_k: int = 0,
                           top_p: float = 0.0,
                           cfg_coef: tp.Optional[float] = None) -> torch.Tensor:
        """Sample next token from the model given a sequence and a set of conditions. The model supports
        multiple sampling strategies (greedy sampling, softmax, top-k, top-p...).

        Args:
            sequence (torch.Tensor): Current sequence of shape [B, K, S]
                with K corresponding to the number of codebooks and S the number of sequence steps.
                S = 1 in streaming mode, except for the first step that contains a bigger prompt.
            condition_tensors (Dict[str, ConditionType): Set of conditions. If CFG is used,
                should be twice the batch size, being the concatenation of the conditions + null conditions.
            use_sampling (bool): Whether to use a sampling strategy or not.
            temp (float): Sampling temperature.
            top_k (int): K for "top-k" sampling.
            top_p (float): P for "top-p" sampling.
            cfg_coef (float): classifier free guidance coefficient
        Returns:
            next_token (torch.Tensor): Next token tensor of shape [B, K, 1].
        """
        B = sequence.shape[0]
        cfg_coef = self.cfg_coef if cfg_coef is None else cfg_coef
        model = self if self._fsdp is None else self._fsdp
        if self.two_step_cfg and cfg_conditions != {}:
            assert isinstance(cfg_conditions, tuple)
            condition_tensors, null_condition_tensors = cfg_conditions
            cond_logits = model(sequence, conditions=[], condition_tensors=condition_tensors)
            state = self.get_streaming_state()
            self.set_streaming_state(unconditional_state)
            uncond_logits = model(sequence, conditions=[], condition_tensors=null_condition_tensors)
            unconditional_state.update(self.get_streaming_state())
            self.set_streaming_state(state)
            logits = uncond_logits + (cond_logits - uncond_logits) * self.cfg_coef
        else:
            assert isinstance(cfg_conditions, dict)
            condition_tensors = cfg_conditions
            if condition_tensors:
                # Preparing for CFG, predicting both conditional and unconditional logits.
                sequence = torch.cat([sequence, sequence], dim=0)
            all_logits = model(
                sequence,
                conditions=[], condition_tensors=condition_tensors)
            if condition_tensors:
                cond_logits, uncond_logits = all_logits.split(B, dim=0)  # [B, K, T, card]
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_coef
            else:
                logits = all_logits

        logits = logits.permute(0, 1, 3, 2)  # [B, K, card, T]
        # logits = logits[..., -1]  # [B x K x card]

        # Apply softmax for sampling if temp > 0. Else, do greedy sampling to avoid zero division error.
        if use_sampling and temp > 0.0:
            probs = torch.softmax(logits / temp, dim=-1)
            if top_p > 0.0:
                next_token = utils.sample_top_p(probs, p=top_p)
            elif top_k > 0:
                next_token = utils.sample_top_k(probs, k=top_k)
            else:
                next_token = utils.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)

        return next_token
    @torch.no_grad()
    def generate_v2a1(
            self, codes: torch.Tensor,
            conditions: tp.List[ConditioningAttributes],
            condition_tensors: tp.Optional[ConditionTensors] = None) -> LMOutput:
        """Given an input tensor of codes [B, K, T] and list of conditions, runs the model
        forward using the specified codes interleaving pattern.

        Args:
            codes (torch.Tensor): Input codes of shape [B, K, T] with B the batch size,
                K the number of codebooks and T the number of timesteps.
            conditions (list[ConditioningAttributes]): conditionings to use when modeling
                the given codes. Note that when evaluating multiple time with the same conditioning
                you should pre-compute those and pass them as `condition_tensors`.
            condition_tensors (dict[str, ConditionType] or None): pre-computed conditioning
                tensors, see `conditions`.
        Returns:
            LMOutput: Language model outputs
                logits (torch.Tensor) of shape [B, K, T, card] corresponding to the provided codes,
                    i.e. the first item corresponds to logits to predict the first code, meaning that
                    no additional shifting of codes and logits is required.
                mask (torch.Tensor) of shape [B, K, T], mask over valid and invalid positions.
                    Given the specified interleaving strategies, parts of the logits and codes should
                    not be considered as valid predictions because of invalid context.
        """
        B, K, T = codes.shape
        device=codes.device
        codes = codes.contiguous()
        # map codes [B, K, T] into pattern sequence [B, K, S] using special_token_id for masked tokens
        pattern = self.pattern_provider.get_pattern(T)
        sequence_codes, sequence_indexes, sequence_mask = pattern.build_pattern_sequence(
            codes, self.special_token_id, keep_only_valid_steps=True
        )
        # apply model on pattern sequence
        model = self if self._fsdp is None else self._fsdp
        logits = model(sequence_codes[:,:,1:], conditions, condition_tensors)  # [B, K, S, card]
        # map back the logits on pattern sequence to logits on original codes: [B, K, S, card] -> [B, K, T, card]
        # and provide the corresponding mask over invalid positions of tokens
        logits = logits.permute(0, 3, 1, 2)  # [B, card, K, S]
        logits=torch.cat([logits,torch.zeros([B,2048,4,1]).to(device)],dim=-1)
        # note: we use nans as special token to make it obvious if we feed unexpected logits
        logits, logits_indexes, logits_mask = pattern.revert_pattern_logits(
            logits, float('nan'), keep_only_valid_steps=True
        )
        logits = logits.permute(0, 2, 3, 1)  # [B, K, T, card]
        logits_mask = logits_mask[None, :, :].expand(B, -1, -1)  # [K, T] -> [B, K, T]
        
        return LMOutput(logits, logits_mask)

    