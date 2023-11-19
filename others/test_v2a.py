import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import torch
from torch import nn
import os
from train import get_emb
import multimodal_whisper
if not os.path.exists('test/v2a'):
    os.mkdir('test/v2a')
cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device)
device='cuda'
model = MusicGen.get_pretrained('small',device=device)
# model.lm.condition_provider.conditioners['description'].output_proj = nn.Linear(in_features=512, out_features=1024, bias=True).cuda()
model.lm.eval()
# model_path='models/v2a_prompt/accompaniment/lm_55999.pt'
# model_path='models/v2a_prompt/promt_304/e_5/lm_29999.pt'
# model_path='models/v2a_prompt/promt_304/5e_6/lm_55999.pt'
model_path='models/v2a_prompt/promt_304/e_6_16/lm_37999.pt'
# # model_path=f"models/2/accompaniment/lm_29999.pt"
state_dict = torch.load(model_path,map_location=device)#13999,61999
model.lm.load_state_dict(state_dict,strict=False)
model.set_generation_params(duration=6,temperature=1,top_p=0.99,use_sampling=True)  # generate 6 seconds.
# model.set_generation_params(duration=6,temperature=1,use_sampling=True)  # generate 6 seconds.

# lrc_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/check/vocal2accompaniment/lrc'
# wav_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/check/vocal2accompaniment/wav'
# lrc_files=os.listdir(lrc_dir)
# wav_files=os.listdir(wav_dir)
# lrc_files.sort()
# wav_files.sort()
# for lrc_file,wav_file in zip(lrc_files,wav_files):
#     vocal_path=wav_dir+'/'+wav_file+'/vocals.wav'
#     lrc_path=lrc_dir+'/'+lrc_file
#     assert wav_file in lrc_file
#     descriptions=[open(lrc_path,'r').read().strip()]
#     wav = model.generate_v2a(vocal_path,descriptions)  # generates 3 samples.
#     audio_write(f'test/v2a/{wav_file}', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

# nohup tar -czvf /data/huangrm/audioLM/raw_data/MusicB.tar.gz /data/huangrm/audioLM/raw_data/MusicB > /data/huangrm/audioLM/raw_data/tar_musicB.log 2>&1 &
# nohup python -u run_v2a_prompt.py > train_v2a_prompt_5e_6.log 2>&1 &
wav_dir='/data/huangrm/audioLM/musicgen_trainer/test/MUSDB18-test/v'
wav_files=os.listdir(wav_dir)
wav_files.sort()
for wav_file in wav_files:
    vocal_path=wav_dir+'/'+wav_file
    descriptions=['伴奏，摇滚：']
    wav = model.generate_v2a_pad6s(vocal_path,descriptions)  # generates 3 samples.
    audio_write(f'test/MUSDB18-test/a/{wav_file}', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)


lrc_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit/train_cutlrc'
wav_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit/train_cutwav'
lrc_files=os.listdir(lrc_dir)
wav_files=os.listdir(wav_dir)
lrc_files.sort()
wav_files.sort()
for lrc_file,wav_file in zip(lrc_files,wav_files):
    vocal_path=wav_dir+'/'+wav_file
    lrc_path=lrc_dir+'/'+lrc_file
    # assert wav_file in lrc_file
    descriptions=['伴奏：'+open(lrc_path,'r').read().strip()]
    wav = model.generate_v2a_pad6s(vocal_path,descriptions)  # generates 3 samples.
    audio_write(f'test/opencpop/a/{wav_file}', wav[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)



