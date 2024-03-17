import os
import librosa
import soundfile as sf
import numpy as np


# Function to peak normalize a wave file
def peak_normalize_audio(audio_path, output_path, target_dbfs=-1.0):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Calculate the peak value
    peak = np.max(np.abs(y))

    # Calculate the desired peak value based on the target dBFS
    target_peak = librosa.db_to_amplitude(target_dbfs)

    # Calculate the gain
    gain = target_peak / peak

    # Apply the gain to the signal
    y_normalized = y * gain

    # Write the normalized audio to the output path
    sf.write(output_path, y_normalized, sr)

# Paths to the source and target folders
source_folder = '/Users/astrid/Documents/Thesis/MEOWS/1everythingLabelledUnpadded'  # Replace with the path to your source folder
target_folder = '/Users/astrid/Documents/Thesis/MEOWS/2peakNormalEverything'  # Replace with the path to your target folder

# Create the target folder if it doesn't exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Normalize each WAV file in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.wav'):
        source_path = os.path.join(source_folder, filename)
        output_path = os.path.join(target_folder, filename)
        peak_normalize_audio(source_path, output_path)
        print(f"Normalized: {filename}")

print("Normalization complete.")
