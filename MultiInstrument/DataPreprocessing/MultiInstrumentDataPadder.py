from pydub import AudioSegment
import os

# Function to find the longest audio file among all instruments
def find_longest_audio(files):
    max_duration = 0
    for file in files:
        audio = AudioSegment.from_file(file)
        duration = len(audio)
        if duration > max_duration:
            max_duration = duration
    return max_duration

# Function to pad audio files in a directory to match the specified length
def pad_audio_in_directory(directory, target_duration):
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            audio = AudioSegment.from_file(file_path)
            if len(audio) < target_duration:
                repetitions = target_duration // len(audio) + 1
                repeated_audio = audio * repetitions
                repeated_audio = repeated_audio[:target_duration]
                # Overwrite the original file with the padded version
                repeated_audio.export(file_path, format="wav")


# Get all audio files across all instrument directories
all_audio_files = []
all_audio_paths = ("MultiInstrument")
all_audio_files.extend([os.path.join(all_audio_paths, file) for file in os.listdir(all_audio_paths) if file.endswith(".wav")])

# Find the longest audio file among all instruments
target_duration = find_longest_audio(all_audio_files)

# Pad audio files in each instrument directory to match the length of the longest audio file
pad_audio_in_directory(all_audio_paths, target_duration)