import torch
import sys
import torchaudio
import torch
sys.path.append('/data/huangrm/audioLM/musicgen_trainer/models/vggish/torchvggish')
from  torchvggish import vggish
import mel_features
import vggish_params 
import vggish_input

project_path='/data/huangrm/audioLM/musicgen_trainer/models/vggish/torchvggish'
state_path='/data/huangrm/audioLM/musicgen_trainer/models/vggish/vggish-10086976.pth'
# model = torch.hub.load(project_path, 'vggish')
# model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model=vggish()
model.eval()

filename='/data/huangrm/audioLM/musicgen_trainer/output_audio.wav'
wav,sr=torchaudio.load(filename)
wav=torch.mean(wav,dim=0)
wav=wav.unsqueeze(0).unsqueeze(0)
examples = vggish_input.wavfile_to_examples(filename)
examples = examples[:,None,:,:] # add dummy dimension for "channel"
examples = torch.from_numpy(examples).float() # Convert input example to float rather than double

embeddings=model.forward(examples)#embeddings.shape torch.Size([3, 128])

print('')

"""
Example for processing a given wav file using torchvggish. Ensure you have the vggish_pca_params.npz file provided by
tensorflow authors, as well as the files provided by audioset.
"""
import sys 
sys.path.append('/data/huangrm/audioLM/musicgen_trainer/models/vggish/torchvggish')
import mel_features
import vggish_params 
import vggish_input
# import vggish_postprocess
import vggish
import torch

# Preprocess input wavfile 
examples = vggish_input.wavfile_to_examples("/data/huangrm/audioLM/musicgen_trainer/bus_chatter.wav")
examples = examples[:,None,:,:] # add dummy dimension for "channel"
examples = torch.from_numpy(examples).float() # Convert input example to float rather than double
# Initialise pretrained vggish & forward pass
net = vggish.pretrained()
embeddings = net.forward(examples)
# import postprocessor
postprocessor = vggish_postprocess.Postprocessor(pca_params_npz_path="./vggish_pca_params.npz")

# postprocess embeddings
embeddings_batch = embeddings.data.numpy()
pca_embeddings = postprocessor.postprocess(embeddings_batch)

pca_embeddings