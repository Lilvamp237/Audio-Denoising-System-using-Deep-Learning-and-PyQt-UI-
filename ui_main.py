import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import glob
from PyQt5.QtCore import Qt, QFile, QTextStream, QTimer
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QFrame, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QPixmap, QIcon, QFont
from utils.audio_utils import plot_waveform, play_audio, stop_audio
from model.denoiser import denoise_audio
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import datetime

class DenoiseApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon("icon.png"))
        self.setWindowTitle("Sup_Audio - Audio Noise Reduction System")
        # Set custom font
        font = QFont("Arial", 16, QFont.Bold)  # Choose font, size, and weight
        self.setFont(font)

        self.setGeometry(400, 200, 1200, 500)  # ⬅️ Bigger window

        self.is_recording = False
        self.recording_data = []  # Will store chunks
        self.sample_rate = 16000  # 16kHz standard

        self.record_timer = QTimer()
        self.record_timer.timeout.connect(self.update_recording_time)
        self.record_elapsed_sec = 0  # Counts how many seconds

        self.record_progress = QProgressBar()
        self.record_progress.setMaximum(30)  # Max 30 seconds
        self.record_progress.setValue(0)
        self.record_progress.setVisible(False)  # Hide until recording starts

        self.file_path = None


        self.upload_btn = QPushButton("Upload Noisy Audio")
        self.upload_btn.clicked.connect(self.load_audio)

        self.start_record_btn = QPushButton("Start Recording")
        self.start_record_btn.clicked.connect(self.start_recording)

        self.stop_record_btn = QPushButton("Stop Recording")
        self.stop_record_btn.clicked.connect(self.stop_recording)
        self.stop_record_btn.setEnabled(False)  # Only enable after starting

        self.denoise_btn = QPushButton("Denoise Audio")
        self.denoise_btn.clicked.connect(self.run_denoising)

        #self.play_noisy_btn = QPushButton("Play Noisy Audio")
        #self.play_noisy_btn.clicked.connect(self.play_noisy_audio)

        #self.play_clean_btn = QPushButton("Play Denoised Audio")
        #self.play_clean_btn.clicked.connect(self.play_clean_audio)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)


        self.status_label = QLabel("Status: Waiting...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setContentsMargins(0, 0, 0, 0)
        self.status_label.setFixedHeight(30)

        # Waveform display areas
        self.noisy_waveform_label = QLabel()
        self.noisy_waveform_label.setFixedSize(550, 250)  # ⬅️ Bigger graph box
        self.noisy_waveform_label.setAlignment(Qt.AlignCenter)
        self.noisy_waveform_label.setStyleSheet("background-color: #2D2545; border: 1px solid #444444;")

        self.denoised_waveform_label = QLabel()
        self.denoised_waveform_label.setFixedSize(550, 250)  # ⬅️ Bigger graph box
        self.denoised_waveform_label.setAlignment(Qt.AlignCenter)
        self.denoised_waveform_label.setStyleSheet("background-color: #2D2545; border: 1px solid #444444;")

        # Layout setup
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)  # Control space *between* buttons
        button_layout.setContentsMargins(0, 0, 0, 0)  # No margins around the button row
        button_layout.addWidget(self.upload_btn)
        #button_layout.addWidget(self.play_noisy_btn)
        #button_layout.addWidget(self.play_clean_btn)
        button_layout.addWidget(self.start_record_btn)
        button_layout.addWidget(self.stop_record_btn)
        button_layout.addWidget(self.denoise_btn)
        button_layout.addWidget(self.exit_btn)

        self.upload_btn.setFixedHeight(40)
        self.denoise_btn.setFixedHeight(40)
        self.exit_btn.setFixedHeight(40)

        waveform_layout = QHBoxLayout()
        waveform_layout.addWidget(self.noisy_waveform_label)
        waveform_layout.addWidget(self.denoised_waveform_label)

        #New buttons
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)

        # Audio player section
        audio_controls_layout = QVBoxLayout()
        self.audio_buttons = []

        for label_text in [
            "❌ Noisy Audio",
            "✅ Clean Audio",
        ]:
            row = QHBoxLayout()
            label = QLabel(label_text)
            label.setFixedWidth(200)
            play_btn = QPushButton("▶️ Play")
            stop_btn = QPushButton("⏹️ Stop")

            
            play_cb, stop_cb = self.make_audio_play_callback(label_text)
            play_btn.clicked.connect(play_cb)
            stop_btn.clicked.connect(stop_cb)
            
            row.addWidget(label)
            row.addWidget(play_btn)
            row.addWidget(stop_btn)
            
            audio_controls_layout.addLayout(row)
            self.audio_buttons.append((label, play_btn, stop_btn))

        main_layout = QVBoxLayout()
        main_layout.setSpacing(5)  # Less vertical space between sections
        main_layout.setContentsMargins(10, 10, 10, 10)  # Small margins around the window
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(self.record_progress)

        main_layout.addLayout(waveform_layout)

        main_layout.addWidget(separator)
        main_layout.addLayout(audio_controls_layout)

        self.setLayout(main_layout)

    def load_audio(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open WAV File", "", "Audio Files (*.wav)")
        if file_path:
            self.file_path = file_path
            self.status_label.setText(f"Loaded: {file_path}")
            plot_waveform(file_path, title="Noisy Audio", output_img="outputs/waveform_noisy.png")
            self.noisy_waveform_label.setPixmap(QPixmap("outputs/waveform_noisy.png").scaled(
                560, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_data = []
            self.record_elapsed_sec = 0

            self.status_label.setText("Recording... Press 'Stop Recording' to finish.")
            self.record_progress.setValue(0)
            self.record_progress.setVisible(True)

            QApplication.processEvents()  # Update UI immediately

            # Start the recording stream
            self.stream = sd.InputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback)
            self.stream.start()
            self.record_timer.start(1000)

            self.start_record_btn.setEnabled(False)
            self.stop_record_btn.setEnabled(True)

    def update_recording_time(self):
        if self.is_recording:
            self.record_elapsed_sec += 1
            self.status_label.setText(f"Recording: {self.record_elapsed_sec} sec")
            self.record_progress.setValue(self.record_elapsed_sec)

            if self.record_elapsed_sec >= 30:
                self.stop_recording()  # Safety auto-stop
                self.show_time_limit_reached_popup()

    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.recording_data.append(indata.copy())
        if sum(len(chunk) for chunk in self.recording_data) >= self.sample_rate * 30:  # 30 sec max
            self.stop_recording()

    def stop_recording(self):
        try:

            if self.is_recording:
                self.is_recording = False

                try:
                    self.record_timer.stop()
                    self.record_progress.setVisible(False)
                    if self.stream:
                        self.stream.stop()
                        self.stream.close()
                except Exception as e:
                    print(f"Error stopping stream: {e}")

                if self.recording_data and len(self.recording_data) > 0:
                    try:
                        audio_array = np.concatenate(self.recording_data, axis=0)
                        # Determine next available recording index
                        existing_recordings = glob.glob("outputs/recorded_*.wav")
                        next_index = len(existing_recordings) + 1
                        noisy_path = f"outputs/recorded_{next_index}.wav"

                        # Save as 16-bit PCM WAV
                        write(noisy_path, self.sample_rate, (audio_array * 32767).astype(np.int16))

                        self.file_path = noisy_path  # Update current file
                        self.status_label.setText(f"Recorded and saved: {noisy_path}")

                        # Plot and update waveform
                        plot_waveform(noisy_path, title="Recorded Noisy Audio", output_img="outputs/waveform_noisy.png")
                        self.noisy_waveform_label.setPixmap(QPixmap("outputs/waveform_noisy.png").scaled(
                            560, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

                        # Auto-denoise
                        self.run_denoising()

                    except Exception as e:
                        print(f"Error during saving or processing recorded audio: {e}")
                        self.status_label.setText("Error saving recording.")
                else:
                    print("No audio recorded.")
                    self.status_label.setText("No audio recorded.")

                self.start_record_btn.setEnabled(True)
                self.stop_record_btn.setEnabled(False)
        except Exception as e:
            print(f"Unhandled error in stop_recording: {e}")

    def show_time_limit_reached_popup(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Recording Limit")
        msg.setText("Maximum recording time (30 sec) reached!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def run_denoising(self):
        if not self.file_path:
            self.status_label.setText("No file loaded!")
            return

        self.status_label.setText("Denoising in progress...")

        existing_outputs = glob.glob("outputs/enhanced_*.wav")
        next_index = len(existing_outputs) + 1

        output_audio_path = f"outputs/enhanced_{next_index}.wav"
        output_noisy_waveform = f"outputs/waveform_noisy_{next_index}.png"
        output_denoised_waveform = f"outputs/waveform_denoised_{next_index}.png"

        # Run denoising
        denoise_audio(self.file_path, output_audio_path)

        # Save waveform plots
        plot_waveform(self.file_path, title="Noisy Audio", output_img=output_noisy_waveform, color="#7F5AF0")
        plot_waveform(output_audio_path, title="Denoised Audio", output_img=output_denoised_waveform, color="#4ADE80")

        self.noisy_waveform_label.setPixmap(QPixmap(output_noisy_waveform).scaled(
            560, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.denoised_waveform_label.setPixmap(QPixmap(output_denoised_waveform).scaled(
            560, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        self.latest_output_audio = output_audio_path
        self.noisy_audio_path = self.file_path
        self.status_label.setText(f"Denoising complete! Saved as enhanced_{next_index}.wav")

    def play_noisy_audio(self):
        if self.file_path and os.path.exists(self.file_path):
            play_audio(self.file_path)
        else:
            self.status_label.setText("No noisy audio to play!")

    def play_clean_audio(self):
        if hasattr(self, "latest_output_audio") and os.path.exists(self.latest_output_audio):
            play_audio(self.latest_output_audio)
        else:
            self.status_label.setText("No denoised audio to play!")

    def make_audio_play_callback(self, label_text):
        def play_callback():
            file_map = {
                "❌ Noisy Audio": getattr(self, "file_path", None),
                "✅ Clean Audio": getattr(self, "latest_output_audio", None),
            }
            path = file_map.get(label_text)
            if path and os.path.exists(path):
                play_audio(path)
            else:
                self.status_label.setText(f"{label_text} not available!")

        def stop_callback():
            stop_audio()  # Assuming you have a `stop_audio()` function in utils.audio_utils
            self.status_label.setText(f"{label_text} stopped.")

        return play_callback, stop_callback
    


def load_qss(file_path):
    with open(file_path, "r") as f:
        return f.read()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DenoiseApp()
    qss = load_qss("dark_purple.qss")  # or dark_blue.qss
    window.setStyleSheet(qss)
    window.show()
    sys.exit(app.exec_())
