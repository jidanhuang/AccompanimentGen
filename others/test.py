import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
from torch import nn
import os
from train import get_emb
import multimodal_whisper
if not os.path.exists('test/test'):
    os.mkdir('test/test')
# cuda_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(cuda_device)
device='cpu'
model = MusicGen.get_pretrained('small',device=device)
# model.lm.condition_provider.conditioners['description'].output_proj = nn.Linear(in_features=512, out_features=1024, bias=True).cuda()
model.lm.eval()
# tokenizer = multimodal_whisper.tokenizer.get_tokenizer(True, language='zh', task='transcribe')
# whisper_model = multimodal_whisper.load_model('base')
# whisper_model.eval()
# /data/huangrm/audioLM/musicgen_trainer/models/2/accompaniment/best/lm_1999.pt
# model_path='/data/huangrm/audioLM/musicgen_trainer/models/accompaniment/wangyitag_lt5/lm_29999.pt'
# model_path='/data/huangrm/audioLM/musicgen_trainer/models/2_conv/accompaniment/lm_15999.pt'
model_path='models/accompaniment/lm_41999.pt'
# # model_path=f"models/2/accompaniment/lm_29999.pt"
state_dict = torch.load(model_path,map_location=device)#13999,61999
model.lm.load_state_dict(state_dict,strict=False)
model.set_generation_params(duration=6,top_k=0)  # generate 6 seconds.

for i in range(3):
    k=int(i*2000)
    # # state_dict.pop('condition_provider.conditioners.description.output_proj.weight', None)
    # # state_dict.pop('condition_provider.conditioners.description.output_proj.bias', None)
    # wav = model.generate_unconditional(4)    # generates 4 unconditional audio samples
    # descriptions = ['钢琴，快乐：我很高兴','伴奏，摇滚：我很高兴','伴奏，治愈：','伴奏，兴奋']#, '如何瞬间冻结时间','记住望着我坚定的双眼','也许已经没有明天',"核反对府邸"]#val,train
    descriptions = ['伴奏，难熬：怎样难熬','伴奏，随风，所欲：随风所欲','伴奏，古风，一城花，浇灌，洛阳：浇灌了洛阳一城花','伴奏，入海，星辰，如：你如星辰入海','伴奏，流行，伤感，脱离，离开，身体：从我身体脱离离开'
                    ,'钢琴，难熬：怎样难熬','钢琴，随风，所欲：随风所欲','钢琴，古风，一城花，浇灌，洛阳：浇灌了洛阳一城花','钢琴，入海，星辰，如：你如星辰入海','钢琴，流行，伤感，脱离，离开，身体：从我身体脱离离开']
    # descriptions = ['Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach',
    #                 'drum and bass beat with intense percussions','drum','guitar','happy']
    # cfg_conditions=get_emb(descriptions,tokenizer,whisper_model,model)
    cfg_conditions=None
    wav = model.generate(descriptions)  # generates 3 samples.

    # melody, sr = torchaudio.load('./assets/bach.mp3')
    # # generates using the melody from the given audio and the provided descriptions.
    # wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'test/test/{i}-{descriptions[idx]}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)