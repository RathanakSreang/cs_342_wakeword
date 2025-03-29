from scipy.io import wavfile
import noisereduce as nr
import numpy as np

def remove_noise(audio_path):
    # load data
    rate, data = wavfile.read(audio_path)
    # perform noise reduction
    data1 = data[:,0]
    data2 = data[0:,1]
    # perform noise reduction
    reduced_noise1 = nr.reduce_noise(y=data1, sr=rate)
    reduced_noise2 = nr.reduce_noise(y=data2, sr=rate)
    reduced_noise = np.stack((reduced_noise1, reduced_noise2), axis=1)
    wavfile.write(audio_path, rate, reduced_noise)


for i in range(100):
    # remove_noise(f"data/recorded/hey_titi/positive/{i}.wav")
    # remove_noise(f"data/recorded/hey_tida/positive/{i}.wav")
    remove_noise(f"data/recorded/negative/{i}.wav")
    # pass
# remove_noise("reduced.wav")