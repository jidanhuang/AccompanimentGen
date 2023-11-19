from BeatNet.BeatNet import BeatNet
import torchaudio
estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
# audio_path='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test/光年-叶琼琳_00:58.74_01:02.25/accompaniment.wav'
# audio_path2='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/test/光年-叶琼琳_00:58.74_01:02.25/vocals.wav'
# audio_path3='/data/huangrm/audioLM/raw_data/cutwav/光年-叶琼琳_00:58.74_01:02.25.wav'
audio_path='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/cutmp3_15s/光年-叶琼琳_1/accompaniment.mp3'
audio_path2='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/cutmp3_15s/光年-叶琼琳_1/vocals.mp3'
audio_path3='/data/huangrm/audioLM/musicgen_trainer/archivo/cutsong/cutwav_15s/光年-叶琼琳_1.wav'
Output = estimator.process(audio_path)#效果也不错
Output2 = estimator.process(audio_path2)#节拍只能读取到较慢的
Output3 = estimator.process(audio_path3)#效果很好
print(Output)#shape:(8, 2)
print(Output2)#shape:(8, 2)
print(Output3)#shape:(8, 2)
wav,sr=torchaudio.load(audio_path)
wav2,sr=torchaudio.load(audio_path2)
wav=wav+wav2
for beat in Output3:
    t=int(beat[0]*sr)
    dt=100
    wav[:,t:t+dt]=2
torchaudio.save('/data/huangrm/audioLM/musicgen_trainer/compare_model/multi_task/beat/beatnet/output_audio1.wav', wav, sr)