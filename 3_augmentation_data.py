from typing import List, Tuple
import audiomentations
import torch_audiomentations
import torchaudio
import torch
import random
from speechbrain.processing.signal_processing import reverberate
import numpy as np
from pathlib import Path
from scipy.io import wavfile
import scipy.signal as sps
import os

def down_sample_rate(clip: str, sr: int = 16000,):
    clip_data, clip_sr = torchaudio.load(clip)
    clip_data = clip_data[0]
    # if clip_data.shape[0] > total_length:
    #     clip_data = clip_data[0:total_length]
    
    # Resample data
    number_of_samples = round(len(clip_data) * float(sr) / clip_sr)
    clip_data = sps.resample(clip_data, number_of_samples)
    return clip_data

def augment_clips(
        clip_paths: List[str],
        sr: int = 16000,
        batch_size: int = 128,
        augmentation_probabilities: dict = {
            "SevenBandParametricEQ": 0.25,
            "TanhDistortion": 0.25,
            "PitchShift": 0.25,
            "BandStopFilter": 0.25,
            "AddColoredNoise": 0.25,
            "AddBackgroundNoise": 0.75,
            "Gain": 1.0,
            "RIR": 0.5
        },
        background_clip_paths: List[str] = [],
        RIR_paths: List[str] = []
        ):
    # First pass augmentations that can't be done as a batch
    augment1 = audiomentations.Compose([
        audiomentations.SevenBandParametricEQ(min_gain_db=-6, max_gain_db=6, p=augmentation_probabilities["SevenBandParametricEQ"]),
        audiomentations.TanhDistortion(
            min_distortion=0.0001,
            max_distortion=0.10,
            p=augmentation_probabilities["TanhDistortion"]
        ),
    ])

    augment2 = torch_audiomentations.Compose([
        torch_audiomentations.PitchShift(
            min_transpose_semitones=-3,
            max_transpose_semitones=3,
            p=augmentation_probabilities["PitchShift"],
            sample_rate=16000,
            mode="per_batch"
        ),
        torch_audiomentations.BandStopFilter(p=augmentation_probabilities["BandStopFilter"], mode="per_batch"),
        torch_audiomentations.AddColoredNoise(
            min_snr_in_db=10, max_snr_in_db=30,
            min_f_decay=-1, max_f_decay=2, p=augmentation_probabilities["AddColoredNoise"],
            mode="per_batch"
        ),
        torch_audiomentations.AddBackgroundNoise(
            p=augmentation_probabilities["AddBackgroundNoise"],
            background_paths=background_clip_paths,
            min_snr_in_db=-10,
            max_snr_in_db=15,
            mode="per_batch"
        ),
        torch_audiomentations.Gain(max_gain_in_db=0, p=augmentation_probabilities["Gain"]),
    ])

    # Iterate through all clips and augment them
    for i in range(0, len(clip_paths), batch_size):
        batch = clip_paths[i:i+batch_size]
        augmented_clips = []
        for clip in batch:
            clip_data = down_sample_rate(clip, sr)

            # Do first pass augmentations
            augmented_clips.append(torch.from_numpy(augment1(samples=clip_data, sample_rate=sr)))

        # Do second pass augmentations
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        augmented_batch = augment2(samples=torch.vstack(augmented_clips).unsqueeze(dim=1).to(device), sample_rate=sr).squeeze(axis=1)

        # Do reverberation
        if augmentation_probabilities["RIR"] >= np.random.random() and RIR_paths != []:
            rir_waveform, sr = torchaudio.load(random.choice(RIR_paths))
            augmented_batch = reverberate(augmented_batch.cpu(), rir_waveform, rescale_amp="avg")

        # yield batch of 16-bit PCM audio data
        yield (augmented_batch.cpu().numpy()*32767).astype(np.int16)

def positive_augment_clips(model, augmentation_rounds = 1, is_original = True, background_paths=[], rir_paths=[]):
    positive_train_output_dir = f"data/recorded/{model}/positive/"
    positive_clips_train = []
    positive_clips_train = [str(i) for i in Path(positive_train_output_dir).glob("*.wav")] * augmentation_rounds
    positive_clips_train_generator = augment_clips(positive_clips_train, batch_size=100,
                                                            background_clip_paths=background_paths,
                                                            RIR_paths=rir_paths)

    if is_original:
        # write orginal to train
        # down sample
        for index, clip in enumerate(Path(positive_train_output_dir).glob("*.wav")):
            rate = 16000
            clip_data = down_sample_rate(clip, rate)
            wavfile.write(
                f"dataset/{model}/positive/origin_{index}.wav", rate=rate, data=clip_data
            )

    for index, clips in enumerate(positive_clips_train_generator):
        for idx, clip in enumerate(clips):
            wavfile.write(
                f"dataset/{model}/positive/{index}_{idx}.wav", rate=16000, data=clip
            )

def negative_augment_clips(model, augmentation_rounds = 1, is_original = True, background_paths=[], rir_paths=[]):
    negative_train_output_dir = f"data/recorded/negative/"
    negative_clips_train = []
    negative_clips_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")] * augmentation_rounds
    
    negative_clips_train_generator = augment_clips(negative_clips_train, batch_size=100,
                                                            background_clip_paths=background_paths,
                                                            RIR_paths=rir_paths)

    if is_original:
        # write orginal to train
        # down sample
        for index, clip in enumerate(Path(negative_train_output_dir).glob("*.wav")):
            rate = 16000
            clip_data = down_sample_rate(clip, rate)
            wavfile.write(
                f"dataset/{model}/negative/origin_{index}.wav", rate=rate, data=clip_data
            )

    for index, clips in enumerate(negative_clips_train_generator):
        for idx, clip in enumerate(clips):
            wavfile.write(
                f"dataset/{model}/negative/{index}_{idx}.wav", rate=16000, data=clip
            )

# The directories containing background sound
background_paths = ['./data/background/audioset_16k', './data/background/resturant', './data/background/cafeshop', './data/background/fma']
# The directories containing Room Impulse Response recordings
rir_paths = [i.path for j in ["./data/background/mit_rirs"] for i in os.scandir(j)]
# pos train data
positive_augment_clips("hey_titi", augmentation_rounds = 10, background_paths=background_paths, rir_paths=rir_paths)
# neg train data
negative_augment_clips("hey_titi", augmentation_rounds = 10, background_paths=background_paths, rir_paths=rir_paths)
