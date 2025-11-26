# 🎵 Sup_Audio - Audio Denoising System

A deep learning-powered audio denoising application with an intuitive PyQt5 GUI. This system uses a U-Net neural network to remove background noise from audio recordings in real-time.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- **🎙️ Real-time Recording**: Record audio directly from your microphone (up to 30 seconds)
- **📂 File Upload**: Load existing `.wav` files for denoising
- **🧠 Deep Learning**: U-Net model trained for audio noise reduction
- **📊 Waveform Visualization**: Compare noisy vs. denoised audio with side-by-side waveforms
- **▶️ Audio Playback**: Play and stop audio files within the application
- **🎨 Dark Theme UI**: Beautiful dark purple/blue themed interface
- **💾 Auto-save**: Automatically saves all recordings and denoised outputs

## 📋 Requirements

### Dependencies

```txt
tensorflow>=2.10.0
keras>=2.10.0
PyQt5>=5.15.0
librosa>=0.10.0
soundfile>=0.12.0
numpy>=1.23.0
matplotlib>=3.5.0
sounddevice>=0.4.6
scipy>=1.9.0
simpleaudio>=1.0.4
```

### System Requirements

- Python 3.8 or higher
- Windows/Linux/macOS
- Microphone (for recording feature)
- 4GB RAM minimum (8GB recommended)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Lilvamp237/Audio-Denoising-System-using-Deep-Learning-and-PyQt-UI-.git
cd Audio-Denoising-System-using-Deep-Learning-and-PyQt-UI-
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Pre-trained Model

Ensure the `denoising_unet_modeltrail322.h5` model file is placed in the `model/` directory.

### 5. Create Output Directory

```bash
mkdir outputs
```

## 🎮 Usage

### Running the Application

```bash
python ui_main.py
```

### Using the Interface

1. **Upload Audio**
   - Click "Upload Noisy Audio" button
   - Select a `.wav` file from your system
   - The noisy waveform will be displayed

2. **Record Audio**
   - Click "Start Recording" to begin recording
   - Speak/play audio (max 30 seconds)
   - Click "Stop Recording" when done
   - Recording auto-saves and denoises automatically

3. **Denoise Audio**
   - Click "Denoise Audio" after loading/recording
   - Wait for processing to complete
   - Denoised waveform appears on the right

4. **Playback**
   - Use ▶️ Play buttons to hear noisy or clean audio
   - Use ⏹️ Stop buttons to stop playback

5. **Change Theme**
   - Edit `ui_main.py` line 327: change `dark_purple.qss` to `dark_blue.qss`

## 📁 Project Structure

```
Audio-Denoising-System/
│
├── model/
│   ├── denoiser.py                          # Denoising logic
│   └── denoising_unet_modeltrail322.h5     # Pre-trained U-Net model
│
├── utils/
│   └── audio_utils.py                       # Audio processing utilities
│
├── outputs/                                 # Auto-generated outputs
│   ├── recorded_*.wav                       # Recorded audio files
│   ├── enhanced_*.wav                       # Denoised audio files
│   └── waveform_*.png                       # Waveform images
│
├── dark_purple.qss                          # Purple theme stylesheet
├── dark_blue.qss                            # Blue theme stylesheet
├── ui_main.py                               # Main application GUI
├── icon.png                                 # Application icon
└── README.md                                # This file
```

## 🧪 Model Details

- **Architecture**: U-Net Convolutional Neural Network
- **Input**: Spectrogram magnitude (513 x 94 x 1)
- **Output**: Enhanced spectrogram magnitude
- **Training**: Trained on noisy audio samples at 16kHz
- **Frame Size**: 94 frames with 1024 FFT, 256 hop length

### Processing Pipeline

1. Load audio at 16kHz sample rate
2. Compute Short-Time Fourier Transform (STFT)
3. Extract magnitude spectrogram
4. Feed through U-Net model in 94-frame chunks
5. Reconstruct audio using Inverse STFT (ISTFT)
6. Save enhanced audio as 16-bit WAV

## 🎨 Themes

The application includes two dark themes:

- **Dark Purple** (`dark_purple.qss`) - Default purple aesthetic
- **Dark Blue** (`dark_blue.qss`) - Alternative blue aesthetic

## 🛠️ Troubleshooting

### Common Issues

**Model File Not Found**
```
Error: No such file or directory: 'denoising_unet_modeltrail322.h5'
```
- Ensure the model file is in the `model/` directory
- Update path in `model/denoiser.py` if needed

**Audio Playback Issues**
```
Error: Cannot play audio
```
- Install `simpleaudio`: `pip install simpleaudio`
- Check system audio drivers

**Recording Not Working**
```
Error: No microphone detected
```
- Grant microphone permissions to Python
- Check default audio input device

**CUDA Warnings** (Optional)
- The app disables GPU by default (`os.environ["CUDA_VISIBLE_DEVICES"] = "-1"`)
- To enable GPU: Comment out line 3 in `ui_main.py`

## 📊 Output Files

All processed files are saved in the `outputs/` directory:

- `recorded_1.wav`, `recorded_2.wav`, ... - Your recordings
- `enhanced_1.wav`, `enhanced_2.wav`, ... - Denoised versions
- `waveform_noisy_*.png` - Noisy waveform plots
- `waveform_denoised_*.png` - Clean waveform plots

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Lilvamp237**

- GitHub: [@Lilvamp237](https://github.com/Lilvamp237)

## 🙏 Acknowledgments

- U-Net architecture inspired by audio denoising research
- PyQt5 for the beautiful GUI framework
- Librosa for powerful audio processing tools
- Community contributions and feedback

## 📧 Contact

Have questions or suggestions? Feel free to open an issue or reach out!

---

**Made with ❤️ and 🎵 by Lilvamp237**
