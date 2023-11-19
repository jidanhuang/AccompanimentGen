


import torchaudio
import torch
import numpy as np
audiopath='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test/光年-叶琼琳_00:58.74_01:02.25/vocals.wav'
# 读取正规化的WAV文件
waveform, sample_rate = torchaudio.load(audiopath)

# 生成白噪音
noise = torch.randn(waveform.shape)

# 调整白噪音的幅度（可以根据需要调整）
noise_amplitude = 0.05 # 调整白噪音的幅度
noisy_waveform = waveform + noise_amplitude * noise

# 保存包含白噪音的音频为新的WAV文件
torchaudio.save('/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test_mp3/add_noise/noisy_audio.wav', noisy_waveform, sample_rate)

print("白噪音已添加到音频文件。")



# from pydub import AudioSegment
# import numpy as np

# # 读取原始MP3文件
# original_audio = AudioSegment.from_mp3('original.mp3')

# # 创建与原始音频相同长度的白噪音
# noise_duration = len(original_audio)
# noise = AudioSegment.silent(duration=noise_duration)

# # 将白噪音添加到原始音频
# noisy_audio = original_audio.overlay(noise)

# # 保存包含白噪音的音频为新的MP3文件
# noisy_audio.export('noisy_original.mp3', format='mp3')

# print("白噪音已添加到音频文件。")
