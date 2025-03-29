import sounddevice as sd
from scipy.io.wavfile import write

def record_audio_and_save(save_path, n_times=50):
    input("To start recording Wake Word press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        input(f"Press to record next or two stop press ctrl + C ({i + 1}/{n_times}): ")

def record_background_sound(save_path, n_times=50):
    input("To start recording your background sounds press Enter: ")
    for i in range(n_times):
        fs = 44100
        seconds = 2 

        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()
        write(save_path + str(i) + ".wav", fs, myrecording)
        print(f"Currently on {i+1}/{n_times}")

# Step 1: Record yourself saying the Wake Word
# - keyword: "ហេ ធីតា"
# print("Recording the Wake Word: ហេ ធីតា\n")
# record_audio_and_save("data/hey_tida/positive/", n_times=100)
# record_audio_and_save("data/recorded/hey_tida/positive/", n_times=100)
# print("Recording the Word Not: ហេ ធីតា\n")
# record_audio_and_save("data/recorded/hey_tida/negative/", n_times=100)
# record_audio_and_save("data/hey_tida/negative/", n_times=100)

# - keyword: "ហេ ទីទី"
# print("Recording the Wake Word: ហេ ទីទី\n")
# record_audio_and_save("data/hey_titi/positive/", n_times=100)
# record_audio_and_save("data/recorded/hey_titi/positive/", n_times=100)
# print("Recording the Word Not: ហេ ទីទី\n")
# record_audio_and_save("data/hey_titi/negative/", n_times=100)

# # Step 2: Record your background sounds (Just let it run, it will automatically record)
# print("Recording the Background sounds:\n")
# record_background_sound("data/background/resturant/", n_times=100)


# print("Recording the Word Not: ហេ ធីតា, ហេ ទីទី\n")
# record_audio_and_save("data/recorded/hey_tida/negative/", n_times=100)

# - n_samples: 10000
# - n_samples_val: 2000
# - n_samples_val: 2000

# Generate words that sound similar ("adversarial") to the input phrase using phoneme overlap
# def generate_adversarial_texts(input_text: str, N: int, include_partial_phrase: float = 0, include_input_words: float = 0):
#     pass

# 1. Generate positive clips for training
# 2. Generate positive clips for testing
# 3. Generate adversarial negative clips for training
# 4. Generate adversarial negative clips for testing

# Do Data Augmentation
# - add background_clip_paths=background_paths,
# - 

# compute_features_from_generator, embbeded: from wav2vec, whisper, ...
