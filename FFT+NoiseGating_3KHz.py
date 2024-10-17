!pip install librosa
!pip install soundfile
!pip install scipy

import numpy as np
import librosa
import soundfile as sf
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

# Step 1: Provide the file path of the noisy audio
audio_path = r'C:\Users\aksha\Deep Learning Labs\MHA Project\Hindi_Audio_Datasets\archive\Sample_911\call_10.mp3'  # Replace this with your actual file path

# Step 2: Load the noisy audio file using librosa
noisy_audio, sr = librosa.load(audio_path, sr=None)  # sr=None preserves the original sample rate

# Step 3: Perform FFT on the noisy audio signal
fft_audio = fft(noisy_audio)

# Step 4: Frequency thresholding (remove frequencies above 3 kHz)
# Get the frequency bins corresponding to the FFT results
frequencies = np.fft.fftfreq(len(fft_audio), 1/sr)

# Apply a threshold to remove noise above 3 kHz
threshold = 3000  # 3 kHz
fft_audio[np.abs(frequencies) > threshold] = 0

# Step 5: Apply inverse FFT to get the filtered audio signal
filtered_audio = np.real(ifft(fft_audio))

# Step 6: Save the filtered audio to a file
output_filename = 'Filtered_Audio.wav'
sf.write(output_filename, filtered_audio, sr)

print(f"Noise reduction using FFT complete! Filtered audio saved as {output_filename}.")

# Optional: Plot the original and filtered signals for visualization
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(noisy_audio)
plt.title('Original Noisy Audio Signal')
plt.subplot(2, 1, 2)
plt.plot(filtered_audio)
plt.title('Filtered Audio Signal (After Frequency Thresholding)')
plt.tight_layout()
plt.show()
