from pydub import AudioSegment
import os


def pad_audio_to_increment(file_path, target_duration_ms):
    audio = AudioSegment.from_file(file_path, format="wav")
    duration_ms = len(audio)
    # Calculate the number of increments needed
    increments = (duration_ms + target_duration_ms - 1) // target_duration_ms
    # Calculate the total duration after padding
    total_duration = increments * target_duration_ms
    # Pad the audio to reach the total duration
    padded_audio = audio + AudioSegment.silent(duration=total_duration - duration_ms)
    return padded_audio


def process_folder(folder_path, output_folder_path, target_duration_ms=1000):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            padded_audio = pad_audio_to_increment(file_path, target_duration_ms)
            output_file_path = os.path.join(output_folder_path, file_name)
            padded_audio.export(output_file_path, format='wav')


# Set your folder paths here
input_folder = '/Users/astrid/Documents/Thesis/MEOWS/justafew'
output_folder = '/Users/astrid/Documents/Thesis/MEOWS/justafewPadded'


# Process the folder
process_folder(input_folder, output_folder)
