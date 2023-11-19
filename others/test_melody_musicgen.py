from audiocraft.models import MusicGen
import torch
# import torchaudio
# def preprocess_audio(audio_path, model: MusicGen, duration: int = 6):
#     wav, sr = torchaudio.load(audio_path)
#     # wav=wav.cpu()
#     # wav=wav.to(torch.float16) 
#     wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
#     wav = wav.mean(dim=0, keepdim=True)
#     end_sample = int(model.sample_rate * duration)
#     if wav.shape[1] <= model.sample_rate * duration:
#         mask_len=model.sample_rate * duration-wav.shape[1]
#         wav = torch.nn.functional.pad(wav, (0,model.sample_rate * duration-wav.shape[1] ), mode='constant', value=0.)
#         mask= torch.ones_like(wav, dtype=torch.bool)
#         mask[:,-mask_len:]=False
#     elif wav.shape[1] > model.sample_rate * duration:
#         wav = wav[:, :end_sample] 
#         mask= torch.ones_like(wav, dtype=torch.bool)
#     assert wav.shape[0] == 1
#     assert wav.shape[1] == model.sample_rate * duration

#     wav = wav
#     wav = wav.unsqueeze(1)
#     return wav,mask
# model = MusicGen.get_pretrained('melody')
# melody_path='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test/光年-叶琼琳_00:58.74_01:02.25/accompaniment.wav'
# melody_wav=preprocess_audio(melody_path,model)
# descriptions = ['伴奏，难熬：怎样难熬','伴奏，随风，所欲：随风所欲','伴奏，古风，一城花，浇灌，洛阳：浇灌了洛阳一城花','伴奏，入海，星辰，如：你如星辰入海','伴奏，流行，伤感，脱离，离开，身体：从我身体脱离离开'
#                 ,'钢琴，难熬：怎样难熬','钢琴，随风，所欲：随风所欲','钢琴，古风，一城花，浇灌，洛阳：浇灌了洛阳一城花','钢琴，入海，星辰，如：你如星辰入海','钢琴，流行，伤感，脱离，离开，身体：从我身体脱离离开']
# # wav = model.generate(descriptions)  # generates 3 samples.
# attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions,prompt=None,melody_wavs=melody_wav)
# model._generate_tokens(attributes, prompt_tokens, False)

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
cuda_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device)
device='cuda'
model = MusicGen.get_pretrained_v2a_melody('small',device=device)
model = MusicGen.get_pretrained('melody',device=device)
model.lm.eval()
# model_path='/data/huangrm/audioLM/musicgen_trainer/models/v2a_prompt/promt_304_vocal_melody/5e_6/lm_1999.pt'
# state_dict = torch.load(model_path,map_location=device)#13999,61999
# model.lm.load_state_dict(state_dict,strict=False)
model.set_generation_params(duration=6)  
#     
    
import os   
# lrc_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit/train_cutlrc'
# wav_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit/train_cutwav'
# lrc_files=os.listdir(lrc_dir)
# wav_files=os.listdir(wav_dir)
# lrc_files.sort()
# wav_files.sort()
# for lrc_file,wav_file in zip(lrc_files,wav_files):
#     vocal_path=wav_dir+'/'+wav_file
#     lrc_path=lrc_dir+'/'+lrc_file
#     # assert wav_file in lrc_file
#     descriptions=[None]
#     melody, sr = torchaudio.load(vocal_path)
#     wav = model.generate_with_chroma(descriptions, melody[None].expand(1, -1, -1), sr)
#     audio_write(f'test/opencpop/a/{wav_file}', wav[0].cpu(), model.sample_rate, strategy="loudness")
    
descriptions = ['happy rock', 'energetic EDM', 'sad cello']
melody_path='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit/train_cutwav/2086003176.wav'
melody, sr = torchaudio.load(melody_path)
# melody, sr = torchaudio.load('/data/huangrm/audioLM/musicgen_trainer/audiocraft_main/assets/bolero_ravel.mp3')
# melody, sr = torchaudio.load('/data/huangrm/audioLM/musicgen_trainer/audiocraft_main/assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)
save_dir='/data/huangrm/audioLM/musicgen_trainer/test/melody/from_vocal'
for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{save_dir}/{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")
