import os
import torch
import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel

# Specify the path to your audio files and the output text file
audio_folder = '/home/pc/Desktop/audio/dir'
output_file = '/home/pc/Desktop/IndicConformer/predicted_transcript.txt'

# Initialize the device and the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncDecCTCModel.restore_from(restore_path='/home/pc/Desktop/IndicConformer/ai4b_indicConformer_hi.nemo')
model.freeze()
model = model.to(device)

# Set the decoder to CTC and RNNT (for both models)
model.cur_decoder = 'ctc'

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Iterate through all audio files in the folder
    for filename in os.listdir(audio_folder):
        # Filter out only audio files (e.g., mp3, wav)
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            file_path = os.path.join(audio_folder, filename)
            
            # Transcribe the audio file
            ctc_text = model.transcribe([file_path], batch_size=1)[0]
            if isinstance(ctc_text, list):
                    ctc_text = ctc_text[0]
            
            # Write the filename and transcription to the output file
            f.write(f"{filename}: {ctc_text}\n")
            print(f"Processed {filename}: {ctc_text}")
    
    print(f"Transcriptions saved to {output_file}")






