import multimodal_whisper
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
model = multimodal_whisper.load_model("base")
result = model.transcribe("audio.mp3")
print(result["text"])