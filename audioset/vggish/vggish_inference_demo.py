# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import numpy as np
import pandas as pd
import six
import soundfile
import tensorflow.compat.v1 as tf
from scipy.io import wavfile

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os
import re
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import seaborn as sns
import librosa


# Directory containing the audio files
audio_dir = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/FINALFINALFINAL/EvertyhingOverlapApril3'

# List of audio files
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]

flags = tf.app.flags


flags.DEFINE_string(
    'wav_folder', 'insert_folder',
    'Path to a wav folder. Should contain signed 16-bit PCM samples. ')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS


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


def extract_pitch_from_filename(filename):
    """
    Extracts the pitch (mean F0) from the filename.

    The function assumes the pitch is at the end of the filename,
    preceded by a hyphen, and followed by the '.wav' extension.

    Parameters:
    - filename: The name of the file, as a string.

    Returns:
    - The extracted pitch as a float, or None if not found.
    """
    # Regex pattern to match a hyphen followed by one or more digits and potentially a decimal point
    pattern = re.compile(r'-(\d+\.\d+|\d+)\.wav$')
    match = pattern.search(filename)
    if match:
        # Convert the matched pitch to a float and return
        return float(match.group(1))
    else:
        # If the pattern is not found, return None
        return None

def main(_):
  #   # Write a WAV of a sine wav into an in-memory file object.
  #   num_secs = 5
  #   freq = 1000
  #   sr = 44100
  #   t = np.arange(0, num_secs, 1 / sr)
  #   x = np.sin(2 * np.pi * freq * t)
  #   # Convert to signed 16-bit samples.
  #   samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
  #   wav_file = six.BytesIO()
  #   soundfile.write(wav_file, samples, sr, format='WAV', subtype='PCM_16')
  #   wav_file.seek(0)

    # # Prepare a postprocessor to munge the model embeddings.
    # pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)
    #
    # # If needed, prepare a record writer to store the postprocessed embeddings.
    # writer = tf.python_io.TFRecordWriter(
    #     FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

    # Add the target variable column (you'll need to determine the logic for setting the target)
    # For example, if you have class labels in the filename:
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)

        # Initialize a list to hold embeddings and labels
        data_list = []

        for audio_file in audio_files:
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

            """
            extract pitch from filename (generated with crepe)
            """
            mean_freq = extract_pitch_from_filename(filename)

            # TODO:
            """       
            supplement VGGish embeddings with other features that might be more 
            discriminative for age classification, like fundamental frequency (pitch), 
            duration, energy distribution (explore), or harmonics.
            
            normalise the necessary, scale were required
            """
            # # Load audio file
            # y, sr = librosa.load(full_audio_path, sr=None)
            #
            # # Extract energy distribution
            # energy = np.mean(librosa.feature.rms(y=y))
            # # Check if standard deviation is zero
            # if np.std(energy) == 0:
            #     energy_normalized = energy  # Keep the original energy value
            # else:
            #     # Normalize energy distribution
            #     energy_normalized = (energy - np.mean(energy)) / np.std(energy)

            # Extract VGGish embeddings
            examples_batch = vggish_input.wavfile_to_examples(full_audio_path)

            # print("examples_batch", examples_batch)

            features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
            embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)

            # Run inference and postprocessing.
            [embedding_batch] = sess.run([embedding_tensor],
                                         feed_dict={features_tensor: examples_batch})

            # print("embedding_batch", embedding_batch)
            print(len(embedding_batch))

            for embedding in embedding_batch:
                data_list.append([embedding, mean_freq, gender_class, target_class, cat_id])

            # # Average the embeddings and append to the list
            # average_embedding = np.mean(embedding_batch, axis=0)
            #
            # # Ensure f0_normalized and energy_normalized are 1D arrays
            # f0_normalized = np.array(f0_normalized)
            # f0_mean = np.mean(f0_normalized)
            # energy_normalized = np.array([energy_normalized])
            #
            # # Concatenate VGGish embeddings with additional features
            # combined_features = np.concatenate([average_embedding, [f0_mean], energy_normalized])

            # Append to the list
            # data_list.append([combined_features, target_class, cat_id])

    # Create a DataFrame with embeddings and corresponding labels
    embeddings_df = pd.DataFrame(data_list, columns=['embedding', 'mean_freq', 'gender', 'target', 'cat_id'])

    # expand 'embedding' column to separate columns
    embeddings_df = pd.concat([pd.DataFrame(embeddings_df['embedding'].tolist()), embeddings_df['mean_freq'],
                               embeddings_df['gender'],
                               embeddings_df['target'],
                               embeddings_df['cat_id']], axis=1)

    # Convert all column names to strings
    embeddings_df.columns = embeddings_df.columns.astype(str)

    # Save the DataFrame to a CSV file
    embeddings_df.to_csv('all_embeddings.csv', index=False)

    # # Perform clustering on the embeddings
    # n_clusters = 4  # Choose the number of clusters
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # embeddings_df['cluster'] = kmeans.fit_predict(embeddings_df.drop(['cat_id', 'target'], axis=1))
    #
    # # Reduce dimensionality for visualization
    # pca = PCA(n_components=2)
    # reduced_embeddings = pca.fit_transform(embeddings_df.drop(['cluster', 'target', 'cat_id'], axis=1))
    # embeddings_df['PCA1'] = reduced_embeddings[:, 0]
    # embeddings_df['PCA2'] = reduced_embeddings[:, 1]
    #
    # # Define target color palette
    # custom_palette = {'KITTEN': 'red', 'YOUNG': 'pink', 'SENIOR': 'blue', 'ADULT': 'purple'}
    #
    # # Visualize the clusters with PCA components
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(x='PCA1', y='PCA2', hue='cluster', data=embeddings_df, palette='viridis', alpha=0.7)
    # plt.title('Cluster Visualization with PCA Components')
    # plt.show()
    #
    # # Overlay the target classes for reference
    # plt.figure(figsize=(10, 8))
    # sns.scatterplot(x='PCA1', y='PCA2', hue='target', data=embeddings_df, palette=custom_palette, alpha=0.7, legend='full')
    # plt.title('Target Class Visualization with PCA Components')
    # plt.show()


        # print(embedding_batch)
        # postprocessed_batch = pproc.postprocess(embedding_batch)
        # print(postprocessed_batch)
        #
        # # For raw embeddings
        # embeddings_df = pd.DataFrame(embedding_batch)
        # embeddings_df.to_csv('raw_embeddings.csv', index=False)
        #
        # # For post-processed embeddings
        # postprocessed_df = pd.DataFrame(postprocessed_batch)
        # postprocessed_df.to_csv('postprocessed_embeddings.csv', index=False)
        #
        # # Write the postprocessed embeddings as a SequenceExample, in a similar
        # # format as the features released in AudioSet. Each row of the batch of
        # # embeddings corresponds to roughly a second of audio (96 10ms frames), and
        # # the rows are written as a sequence of bytes-valued features, where each
        # # feature value contains the 128 bytes of the whitened quantized embedding.
        # seq_example = tf.train.SequenceExample(
        #     feature_lists=tf.train.FeatureLists(
        #         feature_list={
        #             vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
        #                 tf.train.FeatureList(
        #                     feature=[
        #                         tf.train.Feature(
        #                             bytes_list=tf.train.BytesList(
        #                                 value=[embedding.tobytes()]))
        #                         for embedding in postprocessed_batch
        #                     ]
        #                 )
        #         }
        #     )
        # )
        # print(seq_example)


    #     if writer:
    #         writer.write(seq_example.SerializeToString())
    #
    # if writer:
    #     writer.close()


if __name__ == '__main__':
    tf.app.run()
