from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn

from .transcribe import transcribe as transcribe_function
from .decoding import detect_language as detect_language_function, decode as decode_function,multimodal_decode0 as multimodal_decode_function0,multimodal_decode1 as multimodal_decode_function1,multimodal_decode as multimodal_decode_function


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):#torch.Size([1, 80, 3000])
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))#torch.Size([1, 512, 3000])
        x = F.gelu(self.conv2(x))#torch.Size([1, 512, 1500])
        x = x.permute(0, 2, 1)#torch.Size([1, 1500, 512])

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)#torch.Size([1, 1500, 512])#截取30s对应的类
        
        # x = (x + sinusoids(x.shape[-2], 512).to(x.device)).to(x.dtype)#torch.Size([1, 1500, 512])

        for block in self.blocks:#12
            x = block(x)#torch.Size([1, 1500, 512])

        x = self.ln_post(x)#torch.Size([1, 1500, 512])
        return x
        #[1,1500,768] stride=2->1500;768encode嵌入大小


class TextCatBertDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,language_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.catfusion=Linear(language_dim+n_state,n_state)
    def forward(self, x: Tensor, xa: Tensor,bert_f, kv_cache: Optional[dict] = None):#dec_input_ids, audio_features
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        #x torch.Size([5, 1]),上一个预测的token_id，topk=5
        #xa torch.Size([5, 1500, 512])
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#4
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]#torch.Size([5, 1, 512])
        x = x.to(xa.dtype)
        x=torch.cat((x,bert_f),dim=-1)
        x=self.catfusion(x)
        # [5,3,768] [beam_size,token_size of decode_input,emb]
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)#torch.Size([5, 1, 512]),5个candi_token的emb，包含音频信息

        x = self.ln(x)#laynorm
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()#torch.Size([5, 1, 51865])
        #[1,1,51865];[batchsize,seq_len of encode_input,vocab_size]
        return logits

class TextCatClipBertDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,image_dim,language_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.catfusion=Linear(image_dim+language_dim+n_state,n_state)
    def forward(self, x: Tensor, xa: Tensor,bert_f,clip_f, kv_cache: Optional[dict] = None):#dec_input_ids, audio_features
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        #x torch.Size([5, 1]),上一个预测的token_id，topk=5
        #xa torch.Size([5, 1500, 512])
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#4
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]#torch.Size([5, 1, 512])
        x = x.to(xa.dtype)
        x=torch.cat((x,bert_f,clip_f),dim=-1)
        x=self.catfusion(x)
        # [5,3,768] [beam_size,token_size of decode_input,emb]
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)#torch.Size([5, 1, 512]),5个candi_token的emb，包含音频信息

        x = self.ln(x)#laynorm
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()#torch.Size([5, 1, 51865])
        #[1,1,51865];[batchsize,seq_len of encode_input,vocab_size]
        return logits

class TextDecoder(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
    def get_emb(self,x:Tensor, kv_cache: Optional[dict] = None):
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#如果是生成第一个token,offset取在转录sot处，否则在新的位置
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]#torch.Size([5, 1, 512])
        # x = self.ln(x)#laynorm
        return x
    # def get_self_at_emb(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):#dec_input_ids, audio_features
    #     offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#如果是生成第一个token,offset取在转录sot处，否则在新的位置
    #     x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]#torch.Size([5, 1, 512])
    #     x = x.to(xa.dtype)
    #     # [5,3,768] [beam_size,token_size of decode_input,emb]
    #     for block in self.blocks:
    #         x = block(x, xa, mask=self.mask, kv_cache=kv_cache)#torch.Size([5, 1, 512]),5个candi_token的emb，包含音频信息

    #     x = self.ln(x)#laynorm
    #     logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()#torch.Size([5, 1, 51865])
    #     #[1,1,51865];[batchsize,seq_len of encode_input,vocab_size]
    #     return logits

    def forward(self, x: Tensor, xa: Tensor, kv_cache: Optional[dict] = None):#dec_input_ids, audio_features
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        # onset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#如果是生成第一个token,offset取在转录sot处，否则在新的位置
        # offset=offset + x.shape[-1] if kv_cache else x.shape[1]-30#如果第一个token,0~
        # px=torch.zeros(x.shape).to(self.positional_embedding.dtype)
        # px[offset : offset + x.shape[-1]]=self.positional_embedding[onset : ]
        #x torch.Size([5, 1]),上一个预测的token_id，topk=5
        #xa torch.Size([5, 1500, 512])
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#如果是生成第一个token,offset取在转录sot处，否则在新的位置
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]#torch.Size([5, 1, 512])
        x = x.to(xa.dtype)
        # [5,3,768] [beam_size,token_size of decode_input,emb]
        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)#torch.Size([5, 1, 512]),5个candi_token的emb，包含音频信息

        x = self.ln(x)#laynorm
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()#torch.Size([5, 1, 51865])
        #[1,1,51865];[batchsize,seq_len of encode_input,vocab_size]
        return logits

#audioattention和text gatefusion交叉
class TextDecoder_gatefusion0(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,patch_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        self.fusion_blocks: Iterable[GateFusion] = nn.ModuleList(
            [GateFusion(patch_dim, n_state) for _ in range(2)]
        )

        
    def forward(self, x: Tensor, xa: Tensor, language_features,kv_cache: Optional[dict] = None):#dec_input_ids, audio_features
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        #x torch.Size([5, 1]),上一个预测的token_id，topk=5
        #xa torch.Size([5, 1500, 512])
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#4
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]#torch.Size([5, 1, 512])
        x = x.to(xa.dtype)
        # [5,3,768] [beam_size,token_size of decode_input,emb]
        j=0
        for i,block in enumerate(self.blocks):
            if i==1 or i==3:
                x=self.fusion_blocks[j](language_features,x)
                j+=1
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)#torch.Size([5, 1, 512]),5个candi_token的emb，包含音频信息

        x = self.ln(x)#laynorm
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()#torch.Size([5, 1, 51865])
        #[1,1,51865];[batchsize,seq_len of encode_input,vocab_size]
        return logits
#

#audioattention和textattention交叉
class TextDecoder_gatefusion(nn.Module):
    def __init__(self, n_vocab: int, n_ctx: int, n_state: int, n_head: int, n_layer: int,language_dim):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, n_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_ctx, n_state))
        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, cross_attention=True) for _ in range(n_layer)]
        )
        self.ln = LayerNorm(n_state)

        mask = torch.empty(n_ctx, n_ctx).fill_(-np.inf).triu_(1)
        self.register_buffer("mask", mask, persistent=False)
        # self.fusion_blocks: Iterable[GateFusion] = nn.ModuleList(
        #     [GateFusion(patch_dim, n_state) for _ in range(2)]
        # )
        self.caption_crossat=ResidualAttentionBlock(n_state, n_head, cross_attention=True)
        self.caption_dense=Linear(language_dim,n_state)
        
    def forward(self, x: Tensor, xa: Tensor, language_features,kv_cache: Optional[dict] = None):#dec_input_ids, audio_features
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_mels, n_audio_ctx)
            the encoded audio features to be attended on
        """
        #x torch.Size([5, 1]),上一个预测的token_id，topk=5
        #xa torch.Size([5, 1500, 512])
        offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0#4
        x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]#torch.Size([5, 1, 512])
        x = x.to(xa.dtype)
        # [5,3,768] [beam_size,token_size of decode_input,emb]
        for i,block in enumerate(self.blocks):
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)#torch.Size([5, 1, 512]),5个candi_token的emb，包含音频信息
        language_features=self.caption_dense(language_features)
        x=self.caption_crossat(x, language_features, mask=self.mask, kv_cache=kv_cache)
        x = self.ln(x)#laynorm
        logits = (x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)).float()#torch.Size([5, 1, 51865])
        #[1,1,51865];[batchsize,seq_len of encode_input,vocab_size]
        return logits


class GateFusion_multihead(nn.Module):
    def __init__(self, patch_dim, d_model,fusion_head):
        super(GateFusion_multihead, self).__init__()
        self.image_dense = nn.Linear(patch_dim, d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=d_model, kdim=d_model, vdim=d_model, num_heads=fusion_head, batch_first=True)
        self.gate_dense =  nn.Linear(2*d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_features, audio_features):
        # image embedding
        # if image_features.dtype==torch.float16:
        #     self.mha_layer = self.mha_layer.half()        
        #     self.image_dense = self.image_dense.half()        
        #     self.gate_dense = self.gate_dense.half()        
        image_embedding = self.image_dense(image_features) #H_vision [8, 100, 256]->[8, 100, 512]图像特征
        #用音频索引图像，长度和文字相同，融合了图像的文字特征，H_image_attn
        image_att, att_score = self.mha_layer(audio_features, image_embedding, image_embedding)#Q,K,V;[8, 1500, 512]
        merge = torch.cat([audio_features, image_att], dim=-1)#H_image_attn+H_audio;[8, 1500, 1024]
        gate = self.sigmoid(self.gate_dense(merge))#[8, 1500, 1024]->[8, 1500, 512]#每个视觉特征的重要性
        fusion_features = (1 - gate) * audio_features + gate * image_att#[8, 1500, 512]放大了特征重要性的差距？
        
        return fusion_features,att_score

class GateFusion(nn.Module):
    def __init__(self, patch_dim, d_model):
        super(GateFusion, self).__init__()
        self.image_dense = nn.Linear(patch_dim, d_model)
        self.mha_layer = torch.nn.MultiheadAttention(embed_dim=d_model, kdim=d_model, vdim=d_model, num_heads=1, batch_first=True)
        self.gate_dense =  nn.Linear(2*d_model, d_model)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_features, audio_features):
        # image embedding
        # if image_features.dtype==torch.float16:
        #     self.mha_layer = self.mha_layer.half()        
        #     self.image_dense = self.image_dense.half()        
        #     self.gate_dense = self.gate_dense.half()        
        image_embedding = self.image_dense(image_features) #H_vision [8, 100, 256]->[8, 100, 512]图像特征
        #用音频索引图像，长度和文字相同，融合了图像的文字特征，H_image_attn
        image_att, att_score = self.mha_layer(audio_features, image_embedding, image_embedding)#Q,K,V;[8, 1500, 512]
        merge = torch.cat([audio_features, image_att], dim=-1)#H_image_attn+H_audio;[8, 1500, 1024]
        gate = self.sigmoid(self.gate_dense(merge))#[8, 1500, 1024]->[8, 1500, 512]#每个视觉特征的重要性
        fusion_features = (1 - gate) * audio_features + gate * image_att#[8, 1500, 512]放大了特征重要性的差距？
        
        return fusion_features,att_score

class Whisper(nn.Module):
    def __init__(self, dims: ModelDimensions):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.decoder(tokens, self.encoder(mel))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function
class Multimodal_Whisper0(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.patch_dim=patch_dim
        self.image_audio_fusion=GateFusion(self.patch_dim, self.dims.n_audio_state)

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        #out.shape:[16, 125, 51865];[batch_size,seq_len,vocab_num(每个seq的每个token的每个词的score)]
        
        #audio image fusion    
        fusion_features=self.image_audio_fusion(image_features, audio_features) 
        
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function


class BertFusionCatWhisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextCatBertDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            language_dim,
        )
        self.language_dim=language_dim
        self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
        # self.ln = LayerNorm(self.language_dim)
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        fusion_features,_=self.language_audio_fusion(language_features, audio_features) 
        return self.decoder(tokens, fusion_features,language_features[:,:1,:].repeat_interleave(tokens.shape[1], dim=-2))

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function

class BertClipCatWhisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,image_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextCatClipBertDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            image_dim,
            language_dim,
        )
        self.language_dim=language_dim
        self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
        # self.ln = LayerNorm(self.language_dim)
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features:torch.Tensor, mel: torch.Tensor, tokens: torch.Tensor,clip2_global_f) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        fusion_features=self.language_audio_fusion(language_features, audio_features) 
        return self.decoder(tokens, fusion_features,language_features[:,:1,:].repeat_interleave(tokens.shape[1], dim=-2),clip2_global_f)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function

class StackLangFusionWhisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim,n_fusionlayer):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.language_dim=language_dim
        self.fusion_blocks: Iterable[GateFusion] = nn.ModuleList(
            [GateFusion(language_dim, self.dims.n_audio_state) for _ in range(n_fusionlayer)]
        )
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        fusion_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        for fusion in self.fusion_blocks:
            fusion_features,_=fusion(language_features, fusion_features) 
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function

class Multimodal_Whisper1(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.language_dim=language_dim
        self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
        # self.ln = LayerNorm(self.language_dim)
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        #out.shape:[16, 125, 51865];[batch_size,seq_len,vocab_num(每个seq的每个token的每个词的score)]
        # language_features=self.ln(language_features)
        # language_features /=language_features.norm(dim=-1,keepdim=True)
        #audio image fusion    
        fusion_features,_=self.language_audio_fusion(language_features, audio_features) 
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function

class BertWhisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim,fusion_head):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.language_dim=language_dim
        self.fusion_head=fusion_head
        self.language_audio_fusion=GateFusion_multihead(self.language_dim, self.dims.n_audio_state,fusion_head)
        # self.ln = LayerNorm(self.language_dim)
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        #out.shape:[16, 125, 51865];[batch_size,seq_len,vocab_num(每个seq的每个token的每个词的score)]
        # language_features=self.ln(language_features)
        # language_features /=language_features.norm(dim=-1,keepdim=True)
        #audio image fusion    
        fusion_features,_=self.language_audio_fusion(language_features, audio_features) 
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function

class ParallelGateWhisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.language_dim=language_dim
        # self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
        self.patch_dim=patch_dim
        # self.image_audio_fusion=GateFusion(self.patch_dim, self.dims.n_audio_state)
        # n_mlp = (self.dims.n_audio_state*2) * 4
        # self.mlp = nn.Sequential(Linear(self.dims.n_audio_state*2, n_mlp), nn.GELU(), Linear(n_mlp, self.dims.n_audio_state))
        # self.image_dense=Linear(self.patch_dim,self.language_dim)
        # self.ln_l=LayerNorm(language_dim)
        # self.ln_v=LayerNorm(language_dim)
        self.image_language_fusion=GateFusion(self.patch_dim, self.language_dim)
  
        self.lv_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    # def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
    #     return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        #out.shape:[16, 125, 51865];[batch_size,seq_len,vocab_num(每个seq的每个token的每个词的score)]
        #audio image fusion    
        # image_features=self.image_dense(image_features)
        language_features=self.image_language_fusion(image_features, language_features) 
        # fusion_features=torch.concat((language_features,language_features),dim=-1)#8, 1500, 1024
        
        fusion_features=self.lv_audio_fusion(language_features, audio_features) 
        # fusion_features1=self.image_audio_fusion(image_features, audio_features) #8,1500,512
        # fusion_features2=self.language_audio_fusion(language_features, audio_features) 
        # fusion_features=torch.concat((fusion_features1,fusion_features2),dim=-1)#8, 1500, 1024
        # fusion_features=self.mlp(fusion_features)
        
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function


class ParallelMlpFusionWhisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.language_dim=language_dim
        self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
        self.patch_dim=patch_dim
        self.image_audio_fusion=GateFusion(self.patch_dim, self.dims.n_audio_state)
        # self.fusion_dense=Linear(self.dims.n_audio_state*3,self.dims.n_audio_state)
        n_mlp = (self.dims.n_audio_state*3) * 4
        self.mlp = nn.Sequential(Linear(self.dims.n_audio_state*3, n_mlp), nn.GELU(), Linear(n_mlp, self.dims.n_audio_state))
        # self.image_dense=Linear(self.patch_dim,self.language_dim)
        # self.ln_l=LayerNorm(language_dim)
        # self.ln_v=LayerNorm(language_dim)
  
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    # def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
    #     return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        fusion_features1,_=self.image_audio_fusion(image_features, audio_features) #8,1500,512
        fusion_features2,_=self.language_audio_fusion(language_features, audio_features) 
        fusion_features=torch.concat((audio_features,fusion_features1,fusion_features2),dim=-1)#8, 1500, 1024
        fusion_features=self.mlp(fusion_features)
        
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function

class ParallelFusionWhisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.language_dim=language_dim
        self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
        self.patch_dim=patch_dim
        self.image_audio_fusion=GateFusion(self.patch_dim, self.dims.n_audio_state)
        self.fusion_dense=Linear(self.dims.n_audio_state*3,self.dims.n_audio_state)
        # n_mlp = (self.dims.n_audio_state*2) * 4
        # self.mlp = nn.Sequential(Linear(self.dims.n_audio_state*2, n_mlp), nn.GELU(), Linear(n_mlp, self.dims.n_audio_state))
        # self.image_dense=Linear(self.patch_dim,self.language_dim)
        # self.ln_l=LayerNorm(language_dim)
        # self.ln_v=LayerNorm(language_dim)
  
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    # def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
    #     return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        fusion_features1,_=self.image_audio_fusion(image_features, audio_features) #8,1500,512
        fusion_features2,_=self.language_audio_fusion(language_features, audio_features) 
        fusion_features=torch.concat((audio_features,fusion_features1,fusion_features2),dim=-1)#8, 1500, 1024
        fusion_features=self.fusion_dense(fusion_features)
        
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function


class Multimodal_Whisper(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
        )
        self.language_dim=language_dim
        self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
        self.patch_dim=patch_dim
        self.image_audio_fusion=GateFusion(self.patch_dim, self.dims.n_audio_state)
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        #out.shape:[16, 125, 51865];[batch_size,seq_len,vocab_num(每个seq的每个token的每个词的score)]
        #audio image fusion    
        fusion_features=self.image_audio_fusion(image_features, audio_features) 
        fusion_features=self.language_audio_fusion(language_features, fusion_features) 
        return self.decoder(tokens, fusion_features)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function



class Multimodal_Whisper_(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder_gatefusion(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            language_dim
        )

    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        return self.decoder(tokens, audio_features,language_features)
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function
class Whisper_Decode_language_attention(nn.Module):    
    def __init__(self, dims: ModelDimensions,patch_dim,language_dim):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
        )
        self.decoder = TextDecoder_gatefusion(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            language_dim
        )
        self.language_dim=language_dim
        self.language_audio_fusion=GateFusion(self.language_dim, self.dims.n_audio_state)
    def embed_audio(self, mel: torch.Tensor):
        return self.encoder(mel)

    def logits(self, tokens: torch.Tensor, audio_features: torch.Tensor):
        return self.decoder(tokens, audio_features)

    def forward(self,language_features,image_features, mel: torch.Tensor, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        audio_features = self.encoder(mel)#encoder输出，[16, 1500, 512];[batch_size=sample_nums,token_size,emb_size]
        fusion_features,_=self.language_audio_fusion(language_features, audio_features) 
        return self.decoder(tokens, fusion_features,language_features)
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_multilingual(self):
        return self.dims.n_vocab == 51865

    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.decoder.positional_embedding.shape[0]:
                cache[module] = output  # save as-is, for the first token or cross attention
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, MultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    detect_language = detect_language_function
    transcribe = transcribe_function
    decode = multimodal_decode_function




