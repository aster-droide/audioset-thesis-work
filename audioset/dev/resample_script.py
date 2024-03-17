import librosa
import soundfile as sf
import os

# Set your original and target directories
original_directory = '/Users/astrid/Documents/Thesis/MEOWS/catmeow/catmeow-dataset-simone'
resampled_directory = '/Users/astrid/Documents/Thesis/MEOWS/catmeow/catmeow-resampled-feb'

# Ensure the target directory exists
os.makedirs(resampled_directory, exist_ok=True)

# Define the target sample rate for VGGish
target_sample_rate = 16000

# Process each file in the original directory
for filename in os.listdir(original_directory):
    if filename.endswith('.wav'):
        original_path = os.path.join(original_directory, filename)
        resampled_path = os.path.join(resampled_directory, filename)

        # Load the original audio file
        audio, sample_rate = librosa.load(original_path, sr=None)

        # If the sample rate is already the target rate, copy the file over without resampling
        if sample_rate == target_sample_rate:
            sf.write(resampled_path, audio, sample_rate)
        else:
            # Resample the audio to the target sample rate
            audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)

            # Save the resampled audio to the target directory
            sf.write(resampled_path, audio_resampled, target_sample_rate)
            print(f"Resampled {filename} from {sample_rate} Hz to {target_sample_rate} Hz")

print("Batch processing complete.")
