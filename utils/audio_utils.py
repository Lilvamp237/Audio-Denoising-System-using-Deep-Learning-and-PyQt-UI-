import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa

play_obj = None

def plot_waveform(audio_path, title="Waveform", output_img="waveform.png", color="#7F5AF0", figsize=(11, 4.5)):
    y, sr = librosa.load(audio_path, sr=None)
    plt.figure(figsize=figsize)
    ax = plt.gca()

    librosa.display.waveshow(y, sr=sr, color=color)

    # Set background and text colors
    ax.set_facecolor("#1E1B2E")
    ax.tick_params(colors="#CCCCCC")        # X and Y ticks
    ax.xaxis.label.set_color("#CCCCCC")     # X label
    ax.yaxis.label.set_color("#CCCCCC")     # Y label
    ax.title.set_color("#BDB2FF")           # Title
    for spine in ax.spines.values():        # Axis borders
        spine.set_color("#888888")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(output_img, facecolor="#1E1B2E")
    plt.close()

def play_audio(audio_path):
    global play_obj
    wave, sr = librosa.load(audio_path, sr=None)
    audio = (wave * 32767).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, sr)
    #play_obj.wait_done()

def stop_audio():
    global play_obj
    if play_obj:
        play_obj.stop()
        print("Audio stopped.")