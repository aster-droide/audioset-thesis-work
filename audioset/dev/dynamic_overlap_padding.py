from pydub import AudioSegment
import os
import math


def dynamic_pad_audio(file_path, target_frame_length_ms, overlap_percentage):
    """
    Dynamically pads the audio file based on the target frame length and overlap,
    ensuring a minimum length of 1 second (1000 ms) for each clip.

    Args:
    - file_path: Path to the input audio file.
    - target_frame_length_ms: Target length of each frame in milliseconds.
    - overlap_percentage: Percentage of overlap between frames.

    Returns:
    - A padded AudioSegment object.
    """
    audio = AudioSegment.from_file(file_path, format="wav")
    original_duration_ms = len(audio)

    # Ensure minimum length of 1 second
    if original_duration_ms < 1000:
        audio += AudioSegment.silent(duration=1000 - original_duration_ms)
        original_duration_ms = 1000  # Update the duration after padding to 1 second

    # Calculate the step size based on the overlap
    step_size_ms = target_frame_length_ms * (1 - overlap_percentage)

    # Calculate the number of steps that fit into the adjusted audio
    steps_in_audio = original_duration_ms / step_size_ms

    # Calculate the target number of steps (rounded up to ensure coverage)
    target_steps = math.ceil(steps_in_audio)

    # Calculate the required duration to fit the target number of steps
    required_duration_ms = target_steps * step_size_ms

    # Calculate the padding needed
    padding_needed_ms = required_duration_ms - original_duration_ms

    # Pad the audio to match the dynamic frame extraction requirements
    padded_audio = audio + AudioSegment.silent(duration=padding_needed_ms)

    return padded_audio


def process_folder_dynamic_padding(input_folder, output_folder, frame_length_ms=1000, overlap_percentage=0.5):
    """
    Processes each .wav file in the input folder, applying dynamic padding to ensure a minimum length of 1 second,
    and saves the result in the output folder.

    Args:
    - input_folder: Folder containing the input .wav files.
    - output_folder: Folder where the padded .wav files will be saved.
    - frame_length_ms: Length of each frame in milliseconds.
    - overlap_percentage: Overlap between frames as a percentage.
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            padded_audio = dynamic_pad_audio(file_path, frame_length_ms, overlap_percentage)
            output_file_path = os.path.join(output_folder, file_name)
            padded_audio.export(output_file_path, format='wav')


# Set your folder paths here
input_folder = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/EverythingPitchMarch11'
output_folder = '/Users/astrid/Documents/Thesis/MEOWS/FreshMeowFolderFeb24/EverythingOverlapMarch11'

# Process the folder with dynamic padding
process_folder_dynamic_padding(input_folder, output_folder)
