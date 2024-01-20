import copy
import sys
import time
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils import *

sys.path.append('./FTANet-melodic/')
from network.ftanet_pytorch import FTAnet

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps:0')
else:
    device = torch.device('cpu')

print("DEVICE used: "+str(device))

ftanet = FTAnet().to(device)
ftanet.load_state_dict(torch.load('./models/FTA-ResNet_best_version.pth'))
# ftanet.load_state_dict(torch.load('/Users/venkatakrishnanvk/FtaResNet_tanpura.pth'))
ftanet.eval()

hop_len = 441
sr = 44100
time_frame = 128

parser = argparse.ArgumentParser(prog='Carnatic Violin Pitch Predictor')
parser.add_argument('location', help = 'Location of the mp3 or wav audio file')
args = parser.parse_args()
loc = args.location

dot = loc[::-1].index('.')
slash = loc[::-1].index('/')
audio, sample_rate = librosa.load(loc, sr = sr)

length = len(audio)//hop_len

print("STEP 1/3: Predicting Pitch...")

with torch.no_grad():
    prediction_pitch = torch.zeros((321, (length//time_frame + 1)*time_frame)).to(device)
    for i in tqdm(range(0, length, time_frame)):
        W, Cen_freq, _ = cfp.cfp_process(y=audio[i*hop_len:(i+time_frame)*hop_len+hop_len//2], sr = sr, hop = hop_len)
        W = np.concatenate((W, np.zeros((3, 320, 128 - W.shape[-1]))), axis=-1) # Padding
        W_norm = gn.std_normalize(W)
        # W_norm = gn.std_normalize(W[:,:,i:i+time_frame])
        w = np.stack((W_norm, W_norm))
        prediction_pitch[:, i:i+time_frame] = ftanet(torch.Tensor(w).to(device))[0][0][0]
        # count+=1

print("Done!")
print("STEP 2/3: Converting values to Hz...")
y_hat = est(prediction_pitch.to('cpu'), Cen_freq, torch.linspace(hop_len/sr, hop_len/sr*((length//time_frame)*time_frame), ((length//time_frame)*time_frame)))

print("Done!")
print("STEP 3/3: Post processing pitch...")
pitch_proc = PitchProcessor()

pitch_values = y_hat[:,1]
# Interpolate gaps shorter than 250ms (Gulati et al, 2016)
pitch_values = pitch_proc.interpolate_below_length(
    arr=pitch_values,
    val=0.0,
)
# Smooth pitch track a bit
pitch_values = pitch_proc.smoothing(pitch_values, sigma=1)

print("Done!")
print("Writing files...")

plt.figure(figsize=(60,15))
plt.plot(y_hat[:,0], pitch_values)
plt.title(loc[-slash : -dot-1])
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.savefig('plots_and_audios/'+loc[-slash : -dot-1]+" sine_wave_plot.png")

rms_en_pitch = rms_energy(audio, frame_length = 2048, hop_length = hop_len)
output = sinewaveSynth(freqs = np.array(pitch_values[:60*sr//hop_len]), amp = 0.1*np.ones_like(np.array(pitch_values[:60*sr//hop_len])), H = hop_len, fs = sample_rate)
output = lowpass_filter(output, cutoff = 1250, fs = sample_rate)

output = output/np.max(output)
output = np.nan_to_num(output)
wavfile.write("plots_and_audios/"+loc[-slash : -dot-1]+" sine_wave_violin.wav", sample_rate, output)

# output += mul_audio*120
# output = output/np.max(output)
# output = np.nan_to_num(output)
# wavfile.write("plots_and_audios/"+loc[-slash : -dot-1]+" sine_wave_violin+orig.wav", sample_rate, output)