# nn.MSELoss()
import sys
sys.path.append('/data/huangrm/audioLM/bark_trainer_hubert/bark-gui/models')
from transformers import Wav2Vec2FeatureExtractor
import torchaudio.transforms as T

from transformers import AutoModel
import torch
from torch import nn
from datasets import load_dataset
import sys
# sys.path.append('/data/huangrm/audioLM/bark_trainer_hubert/bark-gui')
from MERT_v0_public.cqt import  CQT
from MERT_v0_public.configuration_MERT import  MERTConfig
# protobuf-3.19.6. -> protobuf-3.9.2 
# loading our model weights
model_path='/data/huangrm/audioLM/bark_trainer_hubert/bark-gui/models/MERT_v0_public'
data_path='/data/huangrm/audioLM/bark_trainer_hubert/bark-gui/models/MERT_v1_330M/librispeech_asr_demo'
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_path,trust_remote_code=True)
# load demo audio and set processor
dataset = load_dataset(data_path, "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
resample_rate = processor.sampling_rate#24000模型需要24000
# make sure the sample_rate aligned
if resample_rate != sampling_rate:
    print(f'setting rate from {sampling_rate} to {resample_rate}')
    resampler = T.Resample(sampling_rate, resample_rate)
else:
    resampler = None

# audio file is decoded on the fly
if resampler is None:
    input_audio = dataset[0]["audio"]["array"]
else:
  input_audio = resampler(torch.from_numpy(dataset[0]["audio"]["array"]))#torch.Size([140520]),5.855s
  
inputs = processor(input_audio, sampling_rate=resample_rate, return_tensors="pt")#torch.Size([140520]),5.855s

#cqt
config=MERTConfig.from_pretrained(model_path)
cqt=CQT(config)
with torch.no_grad():
    outputs = cqt(**inputs, output_hidden_states=True)
print(outputs)