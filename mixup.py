import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import os
import random

import random
dir_path = "E:\Kaggle\\birdclef-2022-main\kaggle\input\\birdclef-2023\\train_audio\\abethr1"
files = os.listdir(dir_path)
import pandas as pd
import librosa

audio1,_ = librosa.load(os.path.join(dir_path,files[0]),sr = 32000)
audio2,_ = librosa.load(os.path.join(dir_path,files[1]),sr = 32000)
if random.random()>0.5:
    audio = np.concatenate((audio1[:int(len(audio1)/2)],audio2[int(len(audio2)/2):]))
    image = librosa.feature.melspectrogram(y=audio, sr=32000, n_mels=256, n_fft=2048, hop_length=512, fmin=16,
                                               fmax=16386, )
    image = librosa.power_to_db(image.astype(np.float32), ref=np.max)

    librosa.display.specshow(image)
    plt.show()




