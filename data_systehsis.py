import librosa
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import soundfile as sf

def data_systehsis(background_audio, event_audio, target_samplerate = 44100, audio_second = 10, range_min = 0.2, range_max = 0.8):
    b_data, b_sample_rate = librosa.load(background_audio, sr = None)
    b_data_resampled = librosa.resample(b_data, orig_sr=b_sample_rate, target_sr=target_samplerate)

    e_data, e_sample_rate = librosa.load(event_audio, sr = None)
    e_data_resampled = librosa.resample(e_data, orig_sr=e_sample_rate, target_sr=target_samplerate)

    b_start_point = random.randint(0, len(b_data_resampled) - audio_second*target_samplerate) 

    if len(e_data_resampled) < target_samplerate * audio_second:
        e_start_point = random.randint(0, target_samplerate*audio_second)
        front_zero_list = np.zeros(e_start_point)
        if e_start_point + len(e_data_resampled) < target_samplerate*audio_second:
            end_zero_list = np.zeros(target_samplerate*audio_second - e_start_point -len(e_data_resampled))
            new_e_data = np.concatenate((front_zero_list, np.array(e_data_resampled), end_zero_list))
        else:
            new_e_data = np.concatenate((front_zero_list, np.array(e_data_resampled)))[:target_samplerate*audio_second]
    else:
        e_start_point = random.randint(0, len(e_data_resampled) - audio_second*target_samplerate)
        new_e_data = e_data_resampled[e_start_point:e_start_point+(audio_second*target_samplerate)]

    random_amp = random.uniform(range_min, range_max)
    synthesis = (1-random_amp)*b_data_resampled[b_start_point:b_start_point+(audio_second*target_samplerate)] + random_amp*new_e_data

    return synthesis

wav_folder = 'Wav'
train_wav_folder = os.path.join(wav_folder, 'train')
val_wav_folder = os.path.join(wav_folder, 'validation')
test_wav_folder = os.path.join(wav_folder, 'test')
os.makedirs(train_wav_folder, exist_ok=True)
os.makedirs(val_wav_folder, exist_ok=True)
os.makedirs(test_wav_folder, exist_ok=True)

img_folder = 'Img'
train_img_folder = os.path.join(img_folder, 'train')
val_img_folder = os.path.join(img_folder, 'validation')
test_img_folder = os.path.join(img_folder, 'test')
os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(val_img_folder, exist_ok=True)
os.makedirs(test_img_folder, exist_ok=True)

n_fft = 2048
hop_length = 512
n_mels = 128
random.seed(10)

background_list = os.listdir('Data/Background')
event_list = os.listdir('Data/Event')


for event_folder in event_list:
    all_event_files = os.listdir(os.path.join('Data/Event', event_folder))
    event_files = random.sample(all_event_files, 400)
    for idx, event_file in enumerate(event_files):
        for background_folder in background_list:
            all_background_files = os.listdir(os.path.join('Data/Background', background_folder))
            background_files = random.sample(all_background_files, 10)
            for background_file in background_files:
                result = data_systehsis(os.path.join('Data/Background',background_folder, background_file), os.path.join('Data/Event',event_folder, event_file)) # sr = 44100, 10초의 array
                if idx < 300:
                    os.makedirs(os.path.join(train_wav_folder, event_folder), exist_ok=True)
                    os.makedirs(os.path.join(train_img_folder, event_folder), exist_ok=True)
                    output_wav_path = os.path.join(train_wav_folder, event_folder, f'{os.path.splitext(event_file)[0]}_{os.path.splitext(background_file)[0]}.wav')
                    output_img_path = os.path.join(train_img_folder, event_folder, f'{os.path.splitext(event_file)[0]}_{os.path.splitext(background_file)[0]}.jpg')
                
                elif idx < 350 :
                    os.makedirs(os.path.join(val_wav_folder, event_folder), exist_ok=True)
                    os.makedirs(os.path.join(val_img_folder, event_folder), exist_ok=True)
                    output_wav_path = os.path.join(val_wav_folder, event_folder, f'{os.path.splitext(event_file)[0]}_{os.path.splitext(background_file)[0]}.wav')
                    output_img_path = os.path.join(val_img_folder, event_folder, f'{os.path.splitext(event_file)[0]}_{os.path.splitext(background_file)[0]}.jpg')
                
                else:
                    os.makedirs(os.path.join(test_wav_folder, event_folder), exist_ok=True)
                    os.makedirs(os.path.join(test_img_folder, event_folder), exist_ok=True)
                    output_wav_path = os.path.join(test_wav_folder, event_folder, f'{os.path.splitext(event_file)[0]}_{os.path.splitext(background_file)[0]}.wav')
                    output_img_path = os.path.join(test_img_folder, event_folder, f'{os.path.splitext(event_file)[0]}_{os.path.splitext(background_file)[0]}.jpg')

                sf.write(output_wav_path, result, 44100)
                S = librosa.feature.melspectrogram(y=result, sr=44100, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
                S_dB = librosa.power_to_db(S, ref=np.max)
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.set_position([0, 0, 1, 1])
                ax.set_axis_off()
                librosa.display.specshow(S_dB, sr=44100, hop_length=hop_length, x_axis='time', y_axis='mel', ax=ax)
                plt.savefig(output_img_path, pad_inches=0)
                plt.close(fig)