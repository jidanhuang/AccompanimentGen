import laion_clap
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt() # download the default pretrained checkpoint.
print(model)
text_data = ["I love the contrastive learning", "I love the pretrain model"] 
text_embed = model.get_text_embedding(text_data, use_tensor=True)
print(text_embed)
print(text_embed.shape)