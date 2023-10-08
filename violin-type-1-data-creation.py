import os
import torch
import torchaudio
import argparse

parser = argparse.ArgumentParser(
                    prog='Violin Type-1 Data Creation',
                    description='Creates a .pt file storing all violin audios as torch Tensors in a list')

parser.add_argument('--location', default='../../Datasets/Carnatic Violin Dataset/')
args = parser.parse_args()
loc = args.location

wavFilePaths = []
ext = ".wav"
for (path, dirs, files) in os.walk(loc):
    for f in files:
        if ext.split('.')[-1].lower() == f.split('.')[-1].lower():
            wavFilePaths.append(path + "/" + f)

violin = [torchaudio.load(i)[0][0] for i in wavFilePaths]
torch.save(violin, './carnatic_violin_dataset.pt')
print("Completed!")