from sklearn.decomposition import NMF

def nmf_denoising(audio_path, output_path, n_components=2):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)
    
    # Compute the magnitude spectrogram
    D = librosa.stft(y)
    magnitude, phase = np.abs(D), np.angle(D)
    
    # Apply NMF to the magnitude spectrogram
    model = NMF(n_components=n_components, init='random', random_state=0)
    W = model.fit_transform(magnitude)
    H = model.components_
    
    # Reconstruct the spectrogram with reduced noise
    D_denoised = np.dot(W, H)
    
    # Inverse STFT to reconstruct the time-domain signal
    y_denoised = librosa.istft(D_denoised * np.exp(1j * phase))
    
    # Save the denoised audio
    librosa.output.write_wav(output_path, y_denoised, sr)
    print(f"Denoised audio saved at {output_path}")

# Example usage:
for filename in os.listdir(input_folder):
    if filename.endswith(".mp3"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + "_nmf_denoised.wav")
        nmf_denoising(input_path, output_path)
