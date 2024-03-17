import librosa
import numpy as np
import os
import re
import pandas as pd


# Function to calculate mean fundamental frequency of an audio file
def extract_mean_f0(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Estimate the fundamental frequency using the pYIN algorithm
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

    # Take the mean of the F0 values, ignoring unvoiced frames
    f0_mean = np.nanmean(f0[voiced_flag])

    return f0_mean


# Path to the folder containing audio files
folder_path = '/Users/astrid/Documents/Thesis/MEOWS/2peakNormalEverything'

# Initialize dictionary to hold F0 values
mean_f0_values = {}
target_class_dict = {}
identifier = {}

# Iterate over files in the folder
for file in os.listdir(folder_path):

    if file.endswith('.wav'):  # Assuming files are in 'wav' format

        # Extract the filename from the full path
        filename = os.path.basename(file)

        # Regex pattern to extract cat identifiers
        pattern = re.compile(r'-(\d{3}[A-Z])')
        # Search for the pattern in the filename
        match = pattern.search(filename)
        if match:
            # Extract cat identifier
            cat_id = match.group(1)
            print(cat_id)
        else:
            raise ValueError("Identifier missing or incorrect: ", filename)

        # Extract the target class from the filename
        target_class = filename.split('_')[0]
        target_class_dict[filename] = target_class

        # set the identifier
        identifier[filename] = cat_id

        file_path = os.path.join(folder_path, file)
        mean_f0 = extract_mean_f0(file_path)
        mean_f0_values[file] = mean_f0

# Now mean_f0_values contains the mean F0 for each file
print(target_class_dict)
print(mean_f0_values)

# Create a DataFrame from the dictionaries
df = pd.DataFrame({'Filename': list(mean_f0_values.keys()),
                   'MeanF0': list(mean_f0_values.values()),
                   'Target': [target_class_dict[file] for file in mean_f0_values.keys()],
                   'cat_id': [identifier[file] for file in mean_f0_values.keys()]})

# Save the DataFrame to a CSV file
csv_path = 'peak_normalised_cat_meows_f0_data.csv'
df.to_csv(csv_path, index=False)
