import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

JOJO_FILE = "audio/jojo.wav"

FRAME_SIZE = 2048
HOP_SIZE = 512


jojo, sr = librosa.load(JOJO_FILE)

S_scale = librosa.stft(jojo, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

print(S_scale.shape)

print(type(S_scale[0][0]))

Y_scale = np.abs(S_scale) ** 2

print(Y_scale.shape)

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

Y_log_scale = librosa.power_to_db(Y_scale)

plot_spectrogram(Y_log_scale, sr, HOP_SIZE, y_axis="log")