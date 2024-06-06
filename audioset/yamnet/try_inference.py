# Copyright 2019 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Inference demo for YAMNet."""
from __future__ import division, print_function

import sys

import numpy as np
import os
import re
import resampy
import soundfile as sf
import pandas as pd
import tensorflow_hub as hub
import tensorflow as tf

import params as yamnet_params
import yamnet as yamnet_model

# Directory containing the audio files
audio_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/YAMNet/April23/Looped'

# List of audio files
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]


def extract_age_from_filename(filename):
    """Extracts the age part from the filename."""
    # Use regex to find the age pattern (e.g., "0.5Y" or "0Y")
    match = re.match(r"(\d+\.?\d*)Y", filename)
    if match:
        # Convert the matched age to a float and return
        return float(match.group(1))
    else:
        # If the pattern is not found, raise an error
        raise ValueError("Age missing or incorrect: ", filename)

def extract_gender_from_filename(filename):
    """Extracts the gender part from the filename."""
    # Adjust regex to find the gender indicator more accurately
    match = re.search(r"([MFX])(?=-|\d|\.wav)", filename)
    if match:
        # Return the matched gender
        return match.group(1)
    else:
        # if gender not documented in filename return 'X' (UNKNOWN)
        return "X"


def main():


    # Initialize a list to hold embeddings and labels
    data_list = []

    for audio_file in audio_files:

        assert audio_file, 'Usage: inference.py <wav file> <wav file> ...'

        params = yamnet_params.Params()
        yamnet = yamnet_model.yamnet_frames_model(params)
        yamnet.load_weights('yamnet.h5')
        yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')


        # Extract the filename from the full path
        filename = os.path.basename(audio_file)

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

        # Extract the age from the filename
        age = extract_age_from_filename(filename)
        print(age)
        target_class = age

        # Extract the gender from the filename
        gender = extract_gender_from_filename(filename)
        print(gender)
        gender_class = gender

        # Full path to the audio file
        full_audio_path = os.path.join(audio_dir, audio_file)




        # Decode the WAV file.
        wav_data, sr = sf.read(full_audio_path, dtype=np.int16)
        assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
        waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
        waveform = waveform.astype('float32')

        # Convert to mono and the sample rate expected by YAMNet.
        if len(waveform.shape) > 1:
          waveform = np.mean(waveform, axis=1)
        if sr != params.sample_rate:
          waveform = resampy.resample(waveform, sr, params.sample_rate)

        # Predict YAMNet classes.
        scores, embeddings, spectrogram = yamnet(waveform)
        # Scores is a matrix of (time_frames, num_classes) classifier scores.
        # Average them along time to get an overall classifier output for the clip.
        prediction = np.mean(scores, axis=0)
        # Report the highest-scoring classes and their scores.
        top5_i = np.argsort(prediction)[::-1][:5]
        print(filename, ':\n' +
              '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                        for i in top5_i))

        for embedding in embeddings.numpy():
            print(embedding)
            data_list.append([embedding, gender_class, target_class, cat_id])

    # Create a DataFrame with embeddings and corresponding labels
    embeddings_df = pd.DataFrame(data_list, columns=['embedding', 'gender', 'target', 'cat_id'])

    # expand 'embedding' column to separate columns
    embeddings_df = pd.concat([pd.DataFrame(embeddings_df['embedding'].tolist()),
                               embeddings_df['gender'],
                               embeddings_df['target'],
                               embeddings_df['cat_id']], axis=1)

    # Convert all column names to strings
    embeddings_df.columns = embeddings_df.columns.astype(str)

    # Save the DataFrame to a CSV file
    embeddings_df.to_csv('all_embeddings_yamnet.csv', index=False)


if __name__ == '__main__':
    main()
