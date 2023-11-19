import soundfile as sf
import librosa
import os
from pydub import AudioSegment

def merge_wav_files(voice_path, accompaniment_path, output_path):
    # Read voice WAV file
    voice_data, voice_samplerate = sf.read(voice_path)
    
    # Read accompaniment WAV file
    accompaniment_data, accompaniment_samplerate = sf.read(accompaniment_path)
    
    # Resample if sample rates differ
    if voice_samplerate != accompaniment_samplerate:
        if voice_samplerate < accompaniment_samplerate:
            voice_data = librosa.resample(voice_data, voice_samplerate, accompaniment_samplerate)
            voice_samplerate = accompaniment_samplerate
        else:
            accompaniment_data = librosa.resample(accompaniment_data, accompaniment_samplerate, voice_samplerate)
            accompaniment_samplerate = voice_samplerate
    
    # Ensure the lengths of two WAV files are the same
    min_length = min(len(voice_data), len(accompaniment_data))
    voice_data = voice_data[:min_length][:,0]
    accompaniment_data = accompaniment_data[:min_length]
    # Merge audio data
    # merged_data = voice_data + accompaniment_data
    merged_data =0.3* voice_data+ accompaniment_data
    
    # Save
    sf.write(output_path, merged_data, voice_samplerate)
    
    print("WAV files merged successfully!")
def merge_wav_files2(voice_path, accompaniment_path, output_path):
    # Read voice WAV file
    voice_data, voice_samplerate = sf.read(voice_path)
    
    # Read accompaniment WAV file
    accompaniment_data, accompaniment_samplerate = sf.read(accompaniment_path)
    
    # Resample if sample rates differ
    if voice_samplerate != accompaniment_samplerate:
        if voice_samplerate < accompaniment_samplerate:
            voice_data = librosa.resample(voice_data, voice_samplerate, accompaniment_samplerate)
            voice_samplerate = accompaniment_samplerate
        else:
            accompaniment_data = librosa.resample(accompaniment_data, accompaniment_samplerate, voice_samplerate)
            accompaniment_samplerate = voice_samplerate
    
    # Ensure the lengths of two WAV files are the same
    min_length = min(len(voice_data), len(accompaniment_data))
    voice_data = voice_data[:min_length]
    accompaniment_data = accompaniment_data[:min_length]
    
    # Merge audio data
    # merged_data =0.3* voice_data[:,0] + accompaniment_data
    merged_data =0.3* voice_data+ accompaniment_data
    
    # Save
    sf.write(output_path, merged_data, voice_samplerate)
    
    print("WAV files merged successfully!")

# 示例用法
# for file in filenames:
#     voice_path = f"{vocal_dir}/{file}/vocals.wav"
#     # accompaniment_path =f"{data_dir}/{file}/accompaniment.wav"
#     accompaniment_path =f"{outdir}/{file}.wav"
#     output_path=f"{outdir}/combine/{file}.wav"
#     merge_wav_files(voice_path, accompaniment_path, output_path)
# filenames=os.listdir(vocal_dir)
# 示例用法
# for file in filenames:
#     voice_path = f"{vocal_dir}/{file}"
#     # accompaniment_path =f"{data_dir}/{file}/accompaniment.wav"
#     accompaniment_path =f"{outdir}/{file}"
#     output_path=f"{outdir}/combine/{file}"
#     if os.path.exists(accompaniment_path):
#         merge_wav_files2(voice_path, accompaniment_path, output_path)

vocal_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/opencpop_overfit/train_cutwav'
accompaniment_dir='/data/huangrm/audioLM/musicgen_trainer/compare_model/test/test_result/opencpop/a/vocal_melody'
outdir='/data/huangrm/audioLM/musicgen_trainer/compare_model/test/test_result/opencpop/va/vocal_melody'
if not os.path.exists(outdir):
    os.mkdir(outdir)
filenames=os.listdir(vocal_dir)
for file in filenames:
    voice_path = f"{vocal_dir}/{file}"
    # accompaniment_path =f"{data_dir}/{file}/accompaniment.wav"
    accompaniment_path =f"{accompaniment_dir}/{file}.wav"
    output_path=f"{outdir}/{file}"
    if os.path.exists(accompaniment_path):
        sound1 = AudioSegment.from_wav(voice_path)
        sound2 = AudioSegment.from_wav(accompaniment_path)

        output = sound1.overlay(sound2)  # 把sound2叠加到sound1上面
        # output = sound1.overlay(sound2,position=5000)  # 把sound2叠加到sound1上面，从第5秒开始叠加
        output.export(output_path, format="wav")  # 保存文件
        print("WAV files merged successfully!")



# vocal_dir='/data/huangrm/audioLM/musicgen_trainer/test/MUSDB18-test/v'
# accompaniment_dir='/data/huangrm/audioLM/musicgen_trainer/test/MUSDB18-test/a'
# outdir='/data/huangrm/audioLM/musicgen_trainer/test/MUSDB18-test/va'
# filenames=os.listdir(vocal_dir)

# for file in filenames:
#     voice_path = f"{vocal_dir}/{file}"
#     # accompaniment_path =f"{data_dir}/{file}/accompaniment.wav"
#     accompaniment_path =f"{accompaniment_dir}/{file}.wav"
#     output_path=f"{outdir}/{file}"

#     sound1 = AudioSegment.from_wav(voice_path)
#     sound2 = AudioSegment.from_wav(accompaniment_path)

#     output = sound1.overlay(sound2)  # 把sound2叠加到sound1上面
#     # output = sound1.overlay(sound2,position=5000)  # 把sound2叠加到sound1上面，从第5秒开始叠加
#     output.export(output_path, format="wav")  # 保存文件
#     print("WAV files merged successfully!")


# vocal_dir='/data/huangrm/audioLM/musicgen_trainer/archivo/accompaniment/check/vocal2accompaniment/wav'
# accompaniment_dir='/data/huangrm/audioLM/musicgen_trainer/test/v2a'
# outdir='/data/huangrm/audioLM/musicgen_trainer/test/v2a/combine'
# filenames=os.listdir(vocal_dir)

# for file in filenames:
#     voice_path = f"{vocal_dir}/{file}/vocals.wav"
#     # accompaniment_path =f"{data_dir}/{file}/accompaniment.wav"
#     accompaniment_path =f"{accompaniment_dir}/{file}.wav"
#     output_path=f"{outdir}/{file}.wav"

#     sound1 = AudioSegment.from_wav(voice_path)
#     sound2 = AudioSegment.from_wav(accompaniment_path)

#     output = sound1.overlay(sound2)  # 把sound2叠加到sound1上面
#     # output = sound1.overlay(sound2,position=5000)  # 把sound2叠加到sound1上面，从第5秒开始叠加
#     output.export(output_path, format="wav")  # 保存文件
#     print("WAV files merged successfully!")


