from pydub import AudioSegment
import os


def trim_audio_files(source_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".wav"):
            source_file_path = os.path.join(source_folder, filename)

            # Load the audio file
            audio = AudioSegment.from_wav(source_file_path)

            # Trim 0.5 seconds from the start and end
            trimmed_audio = audio[500:-500]  # 500 milliseconds = 0.5 seconds

            # Save the trimmed audio in the destination folder
            trimmed_file_path = os.path.join(destination_folder, filename)
            trimmed_audio.export(trimmed_file_path, format="wav")

            print(f"Trimmed audio saved as: {trimmed_file_path}")


# Replace the paths with the appropriate folder paths on your system
source_folder = '/Users/astrid/Documents/Thesis/MEOWS/catmeow/catmeow-dataset-simone'
destination_folder = '/Users/astrid/Documents/Thesis/MEOWS/catmeow/catmeow-dataset-simone-trimmed-as-is'
trim_audio_files(source_folder, destination_folder)
