import sys
import argparse
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile

import torch
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
    prediction_pitch = torch.zeros(321, length).to(device)
    for i in tqdm(range(0, length, time_frame)):
        W, Cen_freq, _ = cfp.cfp_process(y=audio[i*hop_len:(i+time_frame)*hop_len+1], sr = sr, hop = hop_len)
        value = W.shape[-1]
        W = np.concatenate((W, np.zeros((3, 320, 128 - W.shape[-1]))), axis=-1) # Padding
        W_norm = gn.std_normalize(W)
        w = np.stack((W_norm, W_norm))
        prediction_pitch[:, i:i+value] = ftanet(torch.Tensor(w).to(device))[0][0][0][:, :value]

print("Done!")
print("STEP 2/3: Converting values to Hz...")

frame_time = hop_len/sr

y_hat = est(
    prediction_pitch.to('cpu'), 
    Cen_freq, 
    torch.linspace(
        frame_time, 
        frame_time*((length//time_frame)*time_frame), 
        ((length//time_frame)*time_frame)
    )
)

print("Done!")
print("STEP 3/3: Post processing pitch...")
pitch_proc = PitchProcessor()

pitch_values = y_hat[:,1]
pitch_values = pitch_proc.interpolate_below_length(         # Interpolate gaps shorter than 250ms (Gulati et al, 2016)
    arr=pitch_values,
    val=0.0,
)
pitch_values = pitch_proc.smoothing(pitch_values, sigma=1)  # Smooth pitch track a bit

print("Done!")
print("Writing files...")

if not os.path.isdir("./result"):
    os.mkdir("./result")
if not os.path.isdir("./result/plots"):
    os.mkdir("./result/plots")
if not os.path.isdir("./result/resynthesized_audios"):
    os.mkdir("./result/resynthesized_audios")

plt.figure(figsize=(60,15))
plt.plot(y_hat[:,0], pitch_values)
plt.title(loc[-slash : -dot-1])
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.xlim(left = 0)
plt.ylim(bottom = 0)
plt.savefig('./result/plots/'+loc[-slash : -dot-1]+"_plot.png")

rms_en_pitch = rms_energy(audio, frame_length = 2048, hop_length = hop_len)
output = sinewaveSynth(freqs = np.array(pitch_values), amp = 0.02*np.ones_like(np.array(pitch_values)), H = hop_len, fs = sample_rate)
output = lowpass_filter(output, cutoff = 1250, fs = sample_rate)

output = output/np.max(output)
output = np.nan_to_num(output)
output = np.concatenate((output, np.zeros(audio.shape[-1] - output.shape[-1])), axis=-1) # Padding)

wavfile.write("result/resynthesized_audios/"+loc[-slash : -dot-1]+"_violin.wav", sample_rate, output)

print("Done! Check results folder for plots and resynthesized audios")