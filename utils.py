import numpy as np
import torch
import torchaudio
import random
import os
import json
import copy
from torch import nn
from torchaudio.transforms import Fade
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from pydub import AudioSegment
import sys
import subprocess

sys.path.append('FTANet-melodic/')
from evaluator import est                   # For estimating frequency from the indices
from evaluator import iseg, melody_eval
import cfp
import generator as gn

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# Taken from MTG/carnatic-pitch-patterns
class PitchProcessor(object):
    def __init__(self, hop_size=441, frame_size=2048, gap_len=25, pitch_preproc=True, voicing=False):

        self.hop_size = hop_size
        self.frame_size = frame_size
        self.gap_len = gap_len

        self.pitch_preproc = pitch_preproc  # Flag for pitch preprocessing
        self.voicing = voicing  # Flag for pitch voicing filtering on audio

        self.sample_rate = 44100  # The standard sampling frequency for Saraga audios

    def pre_processing(self, audio, extracted_pitch):
        # Load audio and adapt pitch length
        extracted_pitch = extracted_pitch[:-2]

        # Zero pad the audio so the length is multiple of 128
        if len(audio) % self.hop_size != 0:
            zero_pad = np.zeros(int((self.hop_size * np.ceil(len(audio) / self.hop_size))) - len(audio))
            audio = np.concatenate([audio, zero_pad])

        # Parsing time stamps and pitch values of the extracted pitch data
        time_stamps = [x[0] for x in extracted_pitch]
        pitch_values = [x[1] for x in extracted_pitch]
        
        # Remove values out of IAM vocal range bounds (Venkataraman et al, 2020)
        pitch_values = self.limiting(
            pitch_values=pitch_values,
            limit_up=1250,
            limit_down=160,
        )

        # To enhance the automatically extracted pitch curve
        if self.pitch_preproc:
            # Interpolate gaps shorter than 250ms (Gulati et al, 2016)
            pitch_values = self.interpolate_below_length(
                arr=pitch_values,
                val=0.0,
            )
            # Smooth pitch track a bit
            pitch_values = self.smoothing(pitch_values, sigma=1)

        # To remove audio content from unvoiced areas
        if self.voicing:
            voiced_samples = []
            for sample in pitch_values:
                if sample > 0.0:
                    voiced_samples = np.concatenate([voiced_samples, self.hop_size * [1]])
                else:
                    voiced_samples = np.concatenate([voiced_samples, self.hop_size * [0]])

            # Set to 0 audio samples which are not voiced while detecting silent zone onsets
            audio_modif = audio.copy()
            silent_zone_on = 1
            silent_onsets = []
            for idx, voiced_sample in enumerate(voiced_samples):
                if voiced_sample == 0:
                    audio_modif[idx] = 0.0
                    if silent_zone_on == 0:
                        silent_onsets.append(idx)
                        silent_zone_on = 1
                else:
                    if silent_zone_on == 1:
                        silent_onsets.append(idx)
                        silent_zone_on = 0

            # Remove first onset if first sample is voiced
            if voiced_samples[0] == 1:
                silent_onsets = silent_onsets[1:] if silent_onsets[0] == 0 else silent_onsets

            # A bit of fade out at sharp gaps
            hop_gap = 16
            for onset in silent_onsets:
                # Make sure that we don't run out of bounds
                if onset + self.hop_size < len(audio_modif):
                    audio_modif[onset - (self.hop_size * hop_gap):onset + (self.hop_size * hop_gap)] = self.smoothing(
                        audio_modif[onset - (self.hop_size * hop_gap):onset + (self.hop_size * hop_gap)], sigma=5
                    )

            return audio_modif, np.array(pitch_values, dtype=np.float64), np.array(time_stamps, dtype=np.float64)

        else:
            return audio, np.array(pitch_values, dtype=np.float64), np.array(time_stamps, dtype=np.float64)

    def interpolate_below_length(self, arr, val):
        """
        Interpolate gaps of value, <val> of
        length equal to or shorter than <gap> in <arr>
        :param arr: Array to interpolate
        :type arr: np.array
        :param val: Value expected in gaps to interpolate
        :type val: number
        :return: interpolated array
        :rtype: np.array
        """
        s = np.copy(arr)
        is_zero = s == val
        cumsum = np.cumsum(is_zero).astype('float')
        diff = np.zeros_like(s)
        diff[~is_zero] = np.diff(cumsum[~is_zero], prepend=0)
        for i, d in enumerate(diff):
            if d <= self.gap_len:
                s[int(i-d):i] = np.nan
        interp = pd.Series(s).interpolate(method='linear', axis=0)\
                             .ffill()\
                             .bfill()\
                             .values
        return interp

    @staticmethod
    def smoothing(pitch_values, sigma=1):
        return gaussian_filter1d(pitch_values, sigma=sigma)
    
    @staticmethod
    def limiting(pitch_values, limit_up, limit_down):
        pitch_values = [x if x < limit_up else 0.0 for x in pitch_values]
        pitch_values = [x if x > limit_down else 0.0 for x in pitch_values]
        return pitch_values

    @staticmethod
    def fix_octave_errors(pitch_track):
        for i in np.arange(len(pitch_track) - 1):
            if (pitch_track[i + 1] != 0) and (pitch_track[i] != 0):
                ratio = pitch_track[i + 1] / pitch_track[i]
                octave_range = np.log10(ratio) / np.log10(2)
                if 0.95 < octave_range < 1.05:
                    pitch_track[i + 1] = pitch_track[i + 1] / 2
    
        return pitch_track

def sinewaveSynth(freqs, amp, H, fs):
	"""
	Synthesis of one sinusoid with time-varying frequency
	freqs, amps: array of frequencies and amplitudes of sinusoids
	H: hop size, 
	fs: sampling rate
	returns y: output array sound
	"""

	t = np.arange(H)/float(fs)                              		# time array
	lastphase = 0                                           		# initialize synthesis phase
	lastfreq = freqs[0]                                     		# initialize synthesis frequency
	y = np.array([])                                        		# initialize output array
	for l in range(freqs.size):                             		# iterate over all frames
		if ((lastfreq==0) and (freqs[l]==0)):                     	# if 0 freq add zeros
			A = np.zeros(H)
			freq1 = np.zeros(H)
		elif ((lastfreq==0) and (freqs[l]>0)):                    	# if starting freq ramp up the amplitude
			A = np.linspace(0, amp[l], H)
			freq1 = np.ones(H)*freqs[l]
		elif ((lastfreq>0) and (freqs[l]>0)):                     	# if freqs in boundaries use both
			A = np.ones(H)*amp[l]
			if (lastfreq==freqs[l]):
				freq1 = np.ones(H)*lastfreq
			else:
				freq1 = np.linspace(lastfreq, freqs[l], H)
		elif ((lastfreq>0) and (freqs[l]==0)):                    	# if ending freq ramp down the amplitude
			A = np.linspace(amp[l],0,H)
			freq1 = np.ones(H)*lastfreq
		else:
			freq1 = np.zeros(H)
			A = np.zeros(H)

		phase = 2*np.pi*freq1*t+lastphase                      		# generate phase values
		yh = A * np.cos(phase) 		                           		# compute sine for one frame
		lastfreq = freqs[l]                                   		# save frequency for phase propagation
		lastphase = np.remainder(phase[H-1], 2*np.pi)         		# save phase to be use for next frame
		y = np.append(y, yh)                                  		# append frame to previous one
	return y

def butter_lowpass(cutoff, fs, slope = 12):
    if(slope%6!=0):
        raise ValueError("Slope needs to be a multiple of 6")
    order = slope//6
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq # normalised cutoff
    b, a = butter(order, normal_cutoff, btype = "low", analog = False)
    return b, a # Coefficients of the filter

def lowpass_filter(data, cutoff, fs, slope = 12):
    b, a = butter_lowpass(cutoff, fs, slope = slope)
    y = filtfilt(b, a, data)
    return y

def rms_energy(signal, frame_length, hop_length):
    return np.array([np.sqrt(np.sum(signal[i:i+frame_length]**2)/frame_length) for i in range(0, len(signal), hop_length)])

def evaluate(ftanet, loader, batch_size, train_data):
    avg_eval_arr = np.array([0, 0, 0, 0, 0], dtype='float64')
    for i, (x, y) in enumerate(loader):
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds = []

        for j in range(num):
            # x: (batch_size, 3, freq_bins, seg_len)
            if j == num - 1:
                X = x[j*batch_size:, :, :, :]
                length = x.shape[0]-j*batch_size
            else:
                X = x[j*batch_size : (j+1)*batch_size, :, :, :]
                length = batch_size
        
            # for k in range(length): # normalization
            #     X[k, :, :, :] = torch.Tensor(gn.std_normalize(X[k, :, :, :].numpy()))

            with torch.no_grad():
                prediction = ftanet(X)[0]
                for pred_batch in prediction:
                    preds.append(pred_batch[0].numpy())
    
        preds = np.array(preds)
        ## (num*bs, freq_bins, seg_len) to (freq_bins, T)
        # preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)

        actual_arr = []
        for y1 in y:
            actual_arr.append(np.array(nn.functional.one_hot(y1, num_classes=train_data[0][0].shape[-2]+1).T))
        actual_arr = np.array(actual_arr)
        actual_arr = iseg(actual_arr)

        CenFreq = train_data.Cen_freq

        time_arr = torch.linspace(train_data.hop/train_data.sr, train_data.hop/train_data.sr*preds.shape[1], preds.shape[1])
        ref_arr = est(actual_arr, CenFreq, time_arr)

        
        # y = torch.stack((time_arr, y.flatten())).T
        # print(y.shape)
        
        # ground-truth
        # ref_arr = y.numpy()
        # time_arr = y.numpy()[:, 0]
        
        # transform to f0ref
        est_arr = est(preds, CenFreq, time_arr)

        # evaluate
        eval_arr = melody_eval(ref_arr, est_arr)
        avg_eval_arr += eval_arr
        if (i+1)%5==0:
            print(avg_eval_arr/(i+1))
    avg_eval_arr /= len(loader)
    # VR, VFA, RPA, RCA, OA
    return avg_eval_arr

class ViolinMixtureDataset(Dataset):
    def __init__(self, root = '../../Datasets/carnatic/violin_solo_dataset/', pure_vio_root = './carnatic_violin_dataset.pt', time_segment = 128, snr: list = [-8, -6.75, -5, -2.5, -1, 0, 2.5, 5], train = True) -> None:
        super(ViolinMixtureDataset, self).__init__()
        self.train = train
        self.sr = 44100
        self.snr = snr
        self.hop = 441 # 10 ms
        self.win_size = 4096 # changed
        self.root = root
        # resample = Resample(44100, 8000)
        violin_file = [root+'violin_solo'+str(i)+'.wav' for i in range(13)]
        # self.violin = [resample(torchaudio.load(i)[0][0]) for i in violin_file]
        self.violin = [torchaudio.load(i, normalize = True)[0][0] for i in violin_file] # List of 1D arrays
        self.violin.extend(torch.load(pure_vio_root))
        loc = root+'vocal/vocalists/'
        # self.vocal = [resample(torchaudio.load(loc+'vocal'+str(i+1)+'.wav')[0][0]) if i > 8 else resample(torchaudio.load(loc+'vocal'+'0'+str(i+1)+'.wav')[0][0]) for i in range(len(os.listdir(loc)))]
        self.vocal = [torchaudio.load(loc+'vocal'+str(i+1)+'.wav', normalize = True)[0][0] if i > 8 else torchaudio.load(loc+'vocal'+'0'+str(i+1)+'.wav', normalize = True)[0][0] for i in range(len(os.listdir(loc)))]
        
        self.pitch_proc = PitchProcessor(voicing = True)
        self.fade = Fade(16, 16)
        self.data = []
        count = False

        try:
            count = False
            self.pitch_tracks = [torch.load(self.root+'pitch_tracks/violin_solo'+str(i)+'_melodia_v3.pt') for i in range(len(self.violin))]
            # self.pitch_tracks = [torch.tensor(np.loadtxt(root+'pitch_tracks/violin_solo'+str(i)+'_melodia_v2.csv', delimiter=',')) for i in range(len(self.violin))]
        except:
            count = True
            self.pitch_tracks = [torch.tensor(np.loadtxt(root+'pitch_tracks/violin_solo'+str(i)+'_melodia_v2.csv', delimiter=',')) for i in range(len(self.violin))]

        if self.train and count == True:
            mixture_loc = self._create_mixtures() # Write the mixtures
            # Read the mixtures and create cfp list
            self._generate_CFP_features(mixture_loc)

        with open("mixtures/cen_freq", "r") as fp:
            self.Cen_freq = json.load(fp)
        
        # Split cfp to chunks
        self.cfp_data = [[] for _ in self.snr] # CHECK!

        self.time = torch.linspace(self.hop/self.sr, self.hop/self.sr*time_segment, time_segment)
        self._extract_CFP(num_examples = len(self.violin))

        temp_pitch_tracks = copy.deepcopy(self.pitch_tracks)

        # Splitting equally among all SNRs
        for i in range(len(self.snr)):
            self.cfp_data[i], self.pitch_tracks = self._split_to_chunks(self.cfp_data[i], temp_pitch_tracks, time_segment)
        self._create_data(self.cfp_data, self.pitch_tracks)
    
    def _create_mixtures(self, gap = 0.5): # As of now it is random singers, change later
        # mixture = []
        mixture_loc = []
        new_pitch_tracks = []
        time_gap = int(gap*self.sr) # 0.5 second
        if not os.path.isdir("./mixtures"):
            subprocess.run(["mkdir","mixtures"])
        for i, (violin_piece, pitch_track) in enumerate(zip(self.violin, self.pitch_tracks)):
            # violin_piece:torch.Tensor = violin_piece/torch.max(violin_piece)
            if not os.path.exists(self.root+'pitch_tracks/violin_solo'+str(i)+'_melodia_v3.pt'):
                # Silences at non-voiced regions as predicted by Melodia
                new_violin_piece, processed_pitch, processed_time_stamps = self.pitch_proc.pre_processing(violin_piece.numpy(), pitch_track.numpy())
                
                # Octave correction
                processed_pitch = PitchProcessor.fix_octave_errors(processed_pitch)

                # Reshape and convert to Tensor
                new_pitch_tracks.append(torch.cat([torch.Tensor(processed_time_stamps).view(-1, 1), torch.Tensor(processed_pitch).view(-1, 1)], 1))
                
                # Save this as .pt file, to avoid re-running
                torch.save(new_pitch_tracks[i], self.root+'pitch_tracks/violin_solo'+str(i)+'_melodia_v3.pt')
                
                # Normalisation
                new_violin_piece = torch.Tensor(new_violin_piece/np.max(new_violin_piece))
                
                # Remove NaN values
                new_violin_piece = torch.nan_to_num(new_violin_piece, nan=0.0)
                
                print(new_pitch_tracks[i].shape)

            for snr_value in self.snr:
                if not os.path.isfile('mixtures/mixture'+str(i)+'_snr'+str(snr_value)+'.wav'):
                    time_point = 0
                    temp_violin_piece = copy.deepcopy(new_violin_piece)
                    while time_point<len(new_violin_piece):
                        vocal_piece = random.sample(self.vocal, k = 1)[0] # Sampling a random vocal snippet
                        while len(vocal_piece) == 0:
                            vocal_piece = random.sample(self.vocal, k = 1)[0]
                        time_point2 = time_point + len(vocal_piece) + time_gap
                        vocal_piece = vocal_piece/torch.max(vocal_piece)  # Normalisation
                        vocal_piece = torch.nan_to_num(vocal_piece, nan=0.0)
                        vocal_piece = self.fade(vocal_piece)
                        if len(temp_violin_piece[time_point:time_point2]) - len(vocal_piece) >= 0:
                            padded_vocal_piece = torch.nn.functional.pad(vocal_piece, (0, len(temp_violin_piece[time_point:time_point2]) - len(vocal_piece)))
                        else:
                            # padded_vocal_piece = vocal_piece[:len(temp_violin_piece[time_point:])] * np.linspace(1.0,0.0,len(temp_violin_piece[time_point:]))
                            padded_vocal_piece = vocal_piece[:len(temp_violin_piece[time_point:time_point2])]
                        # temp_violin_piece[time_point:] = temp_violin_piece[time_point:]+padded_vocal_piece
                        temp_violin_piece[time_point:time_point2] = self.get_mix(snr_value, temp_violin_piece[time_point:time_point2], padded_vocal_piece)
                        time_point = time_point2
                    # mixture.append(temp_violin_piece)
                    # wavfile.write('mixtures/mixture'+str(i)+'.wav', rate = self.sr, data = temp_violin_piece.numpy())
                    temp_violin_piece = temp_violin_piece/torch.max(temp_violin_piece)
                    temp_violin_piece = torch.nan_to_num(temp_violin_piece, nan=0.0)
                    wavfile.write('mixtures/mixture'+str(i)+'_snr'+str(snr_value)+'.wav', rate = self.sr, data = temp_violin_piece.numpy())
                    # # torch.save(temp_violin_piece, 'mixtures/mixture'+str(i)+'.pt')

                mixture_loc.append('mixtures/mixture'+str(i)+'_snr'+str(snr_value)+'.wav') 
        # if not existence:
        #     self.pitch_tracks = new_pitch_tracks
        del new_pitch_tracks     
        return mixture_loc
    
    def _generate_CFP_features(self, mixture_loc):
        # cfp_data = [[] for _ in self.snr]
        for i, loc in enumerate(mixture_loc):
            if not os.path.isfile('mixtures/mixture_CFP'+str(i//len(self.snr))+'_snr'+str(self.snr[i%len(self.snr)])+'.pt'):
                W, Cen_freq, _ = cfp.cfp_process(loc, sr = self.sr, hop = self.hop, win=self.win_size)
                # W_norm = W
                W_norm = gn.std_normalize(W)
                # Shape: (3, freq_bin, time_frames)
                # cfp_data[i%len(self.snr)].append(torch.Tensor(W_norm))

                torch.save(torch.Tensor(W_norm), 'mixtures/mixture_CFP'+str(i//len(self.snr))+'_snr'+str(self.snr[i%len(self.snr)])+'.pt')
        
                with open("mixtures/cen_freq", "w") as fp:
                    json.dump(Cen_freq, fp)
    
    def _extract_CFP(self, num_examples):
        for j in ['mixtures/mixture_CFP'+str(i)+'_snr' for i in range(num_examples)]:
            for idx, k in enumerate(self.snr):
                self.cfp_data[idx].append(torch.load(j+str(k)+'.pt'))
    
    def _split_to_chunks(self, data, pitch_tracks, time_segment = 128, hop_len = 32):
        temp_mixtures = []
        new_pitch_tracks = []
        for cfp_mixture, pitch_track in zip(data, pitch_tracks):
            mixture1 = cfp_mixture # Shape: (3, freq_bin, time_frames)
            pitch_track1:torch.Tensor = pitch_track.T[1]
            temp_pitch_track = torch.zeros_like(pitch_track1, dtype=torch.int64)
            for i, pitch_value in enumerate(pitch_track1):
                ind = cfp.freq2ind(
                    pitch_value.numpy(),
                    StartFreq = 31,
                    StopFreq = 1250,
                    NumPerOct = 60
                )
                if ind == None:
                    ind = 0
                temp_pitch_track[i] = int(ind)
            pitch_track1 = temp_pitch_track
            while True:
                if mixture1.shape[-1]>=time_segment: # Check if longer than 30s
                    temp_mixtures.append(mixture1[:,:,:time_segment])
                    new_pitch_tracks.append(pitch_track1[:time_segment])
                else:
                    mixture1 = self._right_pad_if_necessary(mixture1, length_req=time_segment)
                    temp_mixtures.append(mixture1)
                    new_pitch_tracks.append(self._right_pad_if_necessary(pitch_track1, length_req=time_segment))
                    break
                mixture1 = mixture1[:,:,time_segment-hop_len:]
                pitch_track1 = pitch_track1[time_segment-hop_len:]
        return temp_mixtures, new_pitch_tracks

    def _right_pad_if_necessary(self, CFP, length_req):
        # waveform.shape --> Tensor --> (no. of samples)
        length_signal = CFP.shape[-1]
        if(length_signal < length_req):
            num_missing_samples = length_req - length_signal
            last_dimension_padding = (0, num_missing_samples)
            CFP = torch.nn.functional.pad(CFP, last_dimension_padding)
        else:
            CFP = CFP[:length_req]
        assert CFP.shape[-1]==length_req
        return CFP

    def _create_data(self, cfp, pitch_tracks):
        # Ensures all same snr will go to train and not to test and vice versa
        for cfps in cfp:
            X_train, X_test, y_train, y_test = train_test_split(cfps, pitch_tracks, test_size=0.1, random_state=seed)
            if self.train:
                for i,j in zip(X_train, y_train):
                    # Only nan removed
                    if j.isnan().sum() == 0 and i.isnan().sum() == 0:
                        self.data.append((i,j))
            else:
                for i,j in zip(X_test, y_test):
                    # Only nan removed
                    if j.isnan().sum() == 0 and i.isnan().sum() == 0:
                        self.data.append((i,j))
        random.shuffle(self.data)
        del self.cfp_data
        del cfp
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    @staticmethod
    def get_mix(SNR_level, x_voc, x_har):
        x_har = x_har/torch.max(x_har)
        x_voc = x_voc/torch.max(x_voc)

        x_har = torch.nan_to_num(x_har, nan=0.0)
        x_voc = torch.nan_to_num(x_voc, nan=0.0)
        
        power1 = torch.mean(x_har**2)
        power2 = torch.mean(x_voc**2)
        snr1 = 10*torch.log10(power1)
        snr2 = 10*torch.log10(power2)

        if snr1.item() > -45 and snr2.item() > -45:
            ratio = (SNR_level+snr2)/snr1
            power1__ = 10**((ratio*snr1)/10)
            alpha = torch.sqrt(power1__/power2)
            sig1 = x_voc*alpha

            power1_ = torch.mean(sig1**2)
            
            snr1 = 10*torch.log10(power1_)
            snr2 = 10*torch.log10(power2)
            # delta_snr = snr1 - snr2
        else:
            print(snr2)
            alpha = 1

        x_mix = x_voc*alpha+x_har
        x_mix = x_mix/torch.max(x_mix)
        x_mix = torch.nan_to_num(x_mix, nan=0.0)
        
        return x_mix
    
def pydub_to_np(audio: AudioSegment) -> (np.ndarray, int):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0]. 
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (
            1 << (8 * audio.sample_width - 1)), audio.frame_rate

