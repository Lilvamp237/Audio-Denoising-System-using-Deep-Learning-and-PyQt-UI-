import librosa
import soundfile as sf
import numpy as np
from keras.models import load_model

# Model loading
model_3 = load_model("denoising_unet_modeltrail322.h5")

# Constants
SR = 16000
N_FFT = 1024
HOP_LENGTH = 256

# Prepare input
def prepare_input_spectrogram(audio, sr, n_fft, hop_length, expected_frames=94):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    mag = np.abs(stft)
    current_frames = mag.shape[1]

    if current_frames > expected_frames:
        mag = mag[:, :expected_frames]  # crop
    elif current_frames < expected_frames:
        pad_width = expected_frames - current_frames
        mag = np.pad(mag, ((0, 0), (0, pad_width)), mode='constant')

    model_input = mag[np.newaxis, :, :, np.newaxis]
    return model_input, stft, mag

# ISTFT with magnitude and phase
def istft_from_magnitude(mag, phase_reference):
    phase_reference = phase_reference[:, :mag.shape[1]]
    if phase_reference.shape[1] < mag.shape[1]:
        padding_width = mag.shape[1] - phase_reference.shape[1]
        phase_reference = np.pad(phase_reference, ((0, 0), (0, padding_width)), mode='constant')
    elif phase_reference.shape[1] > mag.shape[1]:
        phase_reference = phase_reference[:, :mag.shape[1]]

    stft_complex = mag * np.exp(1j * np.angle(phase_reference))
    return librosa.istft(stft_complex, hop_length=HOP_LENGTH)

# Main denoising logic
def denoise_audio(input_path, output_path):
    noisy_audio, _ = librosa.load(input_path, sr=SR)

    stft = librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    mag = np.abs(stft)
    phase = stft

    frame_size = 94
    total_frames = mag.shape[1]

    enhanced_mag = np.zeros_like(mag)

    for start in range(0, total_frames, frame_size):
        end = min(start + frame_size, total_frames)
        mag_chunk = mag[:, start:end]

        # Pad if chunk is shorter than frame_size
        if mag_chunk.shape[1] < frame_size:
            pad_width = frame_size - mag_chunk.shape[1]
            mag_chunk = np.pad(mag_chunk, ((0, 0), (0, pad_width)), mode='constant')

        model_input = mag_chunk[np.newaxis, :, :, np.newaxis]
        predicted_chunk = model_3.predict(model_input)[0].squeeze()

        predicted_chunk = predicted_chunk[:, :end - start]  # Crop back if padded
        enhanced_mag[:, start:end] = predicted_chunk

    predicted_audio = istft_from_magnitude(enhanced_mag, phase)
    sf.write(output_path, predicted_audio, SR)
