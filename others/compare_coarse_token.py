import os
import torch
from audiocraft.data.audio import audio_write

#cuda:0能用
cuda_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(cuda_device)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# gpu_id=0
import sys
import torchaudio
# XFORMERS_MORE_DETAILS=1
from audiocraft.models import MusicGen
import torch
import os
def false_portion(tensor):
    num_true = torch.sum(tensor)
    num_elements = tensor.numel()
    return num_true/num_elements

def audio2wav(audio_path, model: MusicGen):
    wav, sr = torchaudio.load(audio_path)
    # wav=wav.cpu()
    # wav=wav.to(torch.float16) 
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
    wav = wav.mean(dim=0, keepdim=True)
    assert wav.shape[0] == 1
    wav = wav.unsqueeze(1)
    return wav
def wav_to_codes(wav,model:MusicGen):
    with torch.no_grad():
        gen_audio = model.compression_model.encode(wav)

    codes, scale = gen_audio

    assert scale is None

    return codes

def vocal_accompaniment_to_codes(vocal_paths,accopaniment_paths,model:MusicGen):
    vs,acs=[],[]
    for vocal_path,accopaniment_path in zip(vocal_paths,accopaniment_paths):
        vocal=audio2wav(vocal_path, model)
        vocal=wav_to_codes(vocal.cuda(),model)
        vs.append(vocal)
        accopaniment=audio2wav(accopaniment_path, model)
        accopaniment=wav_to_codes(accopaniment.cuda(),model)
        acs.append(accopaniment)
    print(false_portion(vs[0]==vs[-1]))
    print(false_portion(acs[0]==acs[-1]))#0.7-0.8左右的相似度。从人耳上判断，MP3音质少了一些细节，但是相差很小
    return vocal,accopaniment
def compress_recover(apath,dpath,model):
    wav=audio2wav(apath, model)
    gen_tokens=wav_to_codes(wav.cuda(),model)
    # generate audio
    assert gen_tokens.dim() == 3
    with torch.no_grad():
        gen_audio = model.compression_model.decode(gen_tokens, None)
    audio_write(dpath, gen_audio[0].cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
model = MusicGen.get_pretrained('small')
wav_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test'
mp3_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test_mp3/mp3_from_test_ffmpeg_mt'
songname=os.listdir(wav_dir)
#1.检查token相同的比例
# for song in songname:
#     vocal_path=wav_dir+'/'+song+'/vocals.wav'    
#     accompaniment_path=wav_dir+'/'+song+'/accompaniment.wav'
#     vocal_path_mp3=mp3_dir+'/'+song+'/vocals.mp3'    
#     accompaniment_path_mp3=mp3_dir+'/'+song+'/accompaniment.mp3'
#     vocal_paths=[vocal_path,vocal_path_mp3]
#     accopaniment_paths=[accompaniment_path,accompaniment_path_mp3]
#     vocal_accompaniment_to_codes(vocal_paths,accopaniment_paths,model)
#2.检查MP3压缩后复原的还原度
# mp3_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test_mp3/mp3_from_test_ffmpeg_mt'
# compress_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test_mp3/compress_mp3'
mp3_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test'
compress_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test_mp3/compress_wav'
songname=os.listdir(wav_dir)
for song in songname:
    vocal_path_mp3=mp3_dir+'/'+song+'/vocals.wav'    
    accompaniment_path_mp3=mp3_dir+'/'+song+'/accompaniment.wav'
    if not os.path.exists(compress_dir+'/'+song):
        os.mkdir(compress_dir+'/'+song)
    vocal_path_compress=compress_dir+'/'+song+'/vocals.wav'    
    accompaniment_path_compress=compress_dir+'/'+song+'/accompaniment.wav'
    compress_recover(vocal_path_mp3,vocal_path_compress,model)    
    compress_recover(accompaniment_path_mp3,accompaniment_path_compress,model)
#MP3:伴奏尚可，人声失真有点严重
#wav:和mp3差不多，可能稍好一些，失真都比较明显