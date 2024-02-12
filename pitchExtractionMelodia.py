# Reference: https://essentia.upf.edu/tutorial_pitch_melody.html

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import essentia.standard as es
import argparse
from mir_eval.sonify import pitch_contour
from tempfile import TemporaryDirectory

parser = argparse.ArgumentParser("Melodia Pitch Extractor",
                                 description="Read Audio, and write melodia output")

parser.add_argument("audioPath")
parser.add_argument("writePath")
parser.add_argument("--sampleRate", default=44100)
parser.add_argument("--frameSize", default=2048)
parser.add_argument("--hopSize", default=128)
parser.add_argument("--format", default='wav')

args = parser.parse_args()

audiofile = args.audioPath
writePath = args.writePath
sr = args.sampleRate
frameSize = args.frameSize
hopSize = args.hopSize
audioFormat = args.format

dot = audiofile[::-1].index('.')
slash = audiofile[::-1].index('/')
audioFolder = audiofile[-slash : -dot-1]

if not os.path.isdir("./result"):
    os.mkdir("./result")
if not os.path.isdir("./result/MELODIA_"+audioFolder):
    os.mkdir("./result/MELODIA_"+audioFolder)

loader = es.EqloudLoader(filename=audiofile, sampleRate=sr)
audio = loader()
print("Duration of the audio sample [sec]:")
print(len(audio)/sr)

# Extract the pitch curve
# PitchMelodia takes the entire audio signal as input (no frame-wise processing is required).

pitch_extractor = es.PredominantPitchMelodia(frameSize=frameSize, hopSize=hopSize)
pitch_values, pitch_confidence = pitch_extractor(audio)

pitch_times = np.linspace(0.0, len(audio)/sr, len(pitch_values))

# Plot the estimated pitch contour and confidence over time.
f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(pitch_times, pitch_values)
axarr[0].set_title('estimated pitch [Hz]')
axarr[1].plot(pitch_times, pitch_confidence)
axarr[1].set_title('pitch confidence')
f.savefig("result/MELODIA_"+audioFolder+"/plot.pdf")

np.savetxt("result/MELODIA_"+audioFolder+"/pitch.txt", np.stack([pitch_times, pitch_values], axis = -1))

# Essentia operates with float32 ndarrays instead of float64, so let's cast it.
synthesized_melody = pitch_contour(pitch_times, pitch_values, sr).astype(np.float32)[:len(audio)]
es.AudioWriter(filename= "./result/MELODIA_" + audioFolder + '/resynth.' + audioFormat, format=audioFormat)(es.StereoMuxer()(audio, synthesized_melody))
