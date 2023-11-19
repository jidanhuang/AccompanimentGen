from pydub import AudioSegment

# 打开.ogg文件
ogg_file = AudioSegment.from_ogg('open_images_train_75faa5c5fd0b4768_21.ogg')

# 进行重采样，并将采样率设为16000 Hz
wav_file = ogg_file.set_frame_rate(16000)

# 将输出保存为.wav文件
output_file = 'open_images_train_75faa5c5fd0b4768_21.wav'
wav_file.export(output_file, format='wav')
