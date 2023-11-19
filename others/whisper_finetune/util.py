import torch
import torchaudio
import tqdm
import io
import librosa
import torch
import torchaudio
import librosa
import torch
import torchaudio
import os
import numpy as np
import multimodal_whisper
from pydub import AudioSegment
import subprocess
import io
import soundfile as sf
import torchaudio
import ffmpeg
import sys
from functools import reduce

from torch.nn.modules.module import _addindent

def load_audio(file: str, sr: int = 16000):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
def load_wav(wav_path, sample_rate:int=16000) -> torch.Tensor:
    """load audio file"""
    waveform, sr = torchaudio.load(wav_path, normalize=True)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform

def load_ogg_as_waveform(ogg_path, sample_rate=16000):
    """load audio file"""
    audio, sr = librosa.load(ogg_path, sr=sample_rate, mono=True)
    return torch.from_numpy(audio)
def load_ogg(ogg_path, sample_rate=16000):
    # Load the .ogg file using Pydub
    sound = AudioSegment.from_file(ogg_path, format="ogg")

    # Convert to raw audio data as bytes
    raw_audio = sound.raw_data

    # Convert raw audio to a PyTorch Tensor
    waveform = torch.from_numpy(np.frombuffer(raw_audio, dtype=np.int16))

    # Reshape and convert to mono audio by taking the mean of the channels
    waveform = waveform.view(-1, sound.channels).float().mean(-1)

    # Resample waveform to the desired sample rate if necessary
    if sound.frame_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sound.frame_rate, new_freq=sample_rate
        )
        waveform = resampler(waveform)

    return waveform

def load_ogg_as_wav(ogg_path, sample_rate=16000):
    """Load an .ogg file as a waveform"""
    audio, sr = sf.read(ogg_path, dtype="float32", always_2d=True)
    if sr != sample_rate:
        audio = torch.tensor(audio).transpose(0, 1)
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio).squeeze()
    else:
        audio = torch.tensor(audio).squeeze()
    return audio




def ogg_to_wav_ffmpeg(in_file, sample_rate=16000):
    cmd = f"ffmpeg -i {in_file} -ar {sample_rate} -ac 1 -f wav pipe:1"
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    waveform, sr = torchaudio.load(io.BytesIO(output), normalize=True)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    return waveform


def load_txt(txt_path) -> str:
    txt = open(txt_path, "r").read()
    txt = txt.replace("\n", "")

    return txt


def load_and_convert_to_mel_use_librosa(ogg_path, duration=30, sample_rate=16000, n_mel_channels=80):
    # Load audio from ogg file
    audio, sr = librosa.load(ogg_path, sr=None, mono=True, duration=duration)#(696960,)
    # Resample if necessary
    if sample_rate != sr:
        audio = librosa.resample(audio, sr, sample_rate)#(232320,)
    # Pad or trim the audio to be exactly the desired duration
    if len(audio) < sample_rate * duration:
        shortage = sample_rate * duration - len(audio)
        audio = np.pad(audio, (0, shortage), mode='constant')
    elif len(audio) > sample_rate * duration:
        audio = audio[:sample_rate * duration]
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    mel_spec = multimodal_whisper.log_mel_spectrogram(audio).numpy().astype(np.float32)#(80, 3000)
    mel_spec = torch.from_numpy(mel_spec.astype(np.float32))

    # Convert to tensor and add batch dimension
    # window = torch.hann_window(400).to(audio.device)

    # # Convert to mel spectrogram
    # mel_transform = torchaudio.transforms.MelSpectrogram(
    #     sample_rate=sample_rate, n_mels=n_mel_channels, n_fft=400, hop_length=160, window_fn=window)
    # mel_spec = mel_transform(audio_tensor)
    # mel_spec=mel_spec[:,:,:3000]
    return mel_spec

    # Convert to mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mel_channels, n_fft=400, hop_length=160, window_fn=window)
    mel_spec = mel_transform(audio_tensor)
    mel_spec=mel_spec[:,:,:3000]
    return mel_spec

    # Convert to mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_mels=n_mel_channels, n_fft=400, hop_length=160, window_fn=window)
    mel_spec = mel_transform(audio_tensor)
    mel_spec=mel_spec[:,:,:3000]
    return mel_spec


def load_and_convert_to_mel(ogg_path, duration=30, sample_rate=16000, n_mel_channels=80):
    # Load audio from ogg file
    wav = load_audio(ogg_path)#
    # Pad or trim the audio to be exactly the desired duration
    # wav = multimodel_whisper.pad_or_trim(wav.flatten())#只截取了30s?shape:torch.Size([2, 571200])->torch.Size([480000])
    mel = multimodal_whisper.log_mel_spectrogram(wav.flatten())#(80, 3000)
    mel = multimodal_whisper.pad_or_trim(mel, 3000)


    return mel
def get_val_list():
    setname='open_images_validation_localized_narratives'
    with open('data_narrative/data_list/'+setname+'.txt', 'r') as f:
        image_ids = f.readlines()
        #移除行末换行符
    data_list = [line.rstrip('\n') for line in image_ids]
    return data_list
def get_one_train_list(i):
    setname=f'open_images_train_v6_localized_narratives-0000{i}-of-00010'
    with open('data_narrative/data_list/'+setname+'.txt', 'r') as f:
        image_ids = f.readlines()
        #移除行末换行符
    data_list = [line.rstrip('\n') for line in image_ids]
    return data_list
def split_dataset(setname):
    with open('data_narrative/data_list/'+setname+'.txt', 'r') as f:
        image_ids = f.readlines()
        #移除行末换行符
    data_list = [line.rstrip('\n') for line in image_ids]
    # decide the ratio of validation and training data
    val_ratio = 0.1 # for example, 20% for validation and 80% for training

    # get the number of validation and training data
    val_num = int(len(data_list) * val_ratio)
    train_num = len(data_list) - val_num

    # split the list accordingly
    val_list = data_list[:val_num]
    train_list = data_list[val_num:]
    
    return train_list,val_list    


def summary(model, file=sys.stdout):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            if hasattr(p, 'shape'):
                total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stdout:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        if isinstance(file, str):
            file = open(file, 'w')
        print(string, file=file)
        file.flush()

    return count



