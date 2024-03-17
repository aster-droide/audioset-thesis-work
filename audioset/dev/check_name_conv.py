import os
import re

def check_filenames(directory):
    # Regular expression pattern to match filenames with the -XXXA format
    # This pattern assumes that the identifier is preceded and followed by other text
    pattern = re.compile(r'.*-\d{3}[A-Z].*')

    # List to store filenames that do not match the pattern
    non_compliant_files = []

    # Iterate through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file matches the pattern
        if not pattern.match(filename):
            non_compliant_files.append(filename)

    return non_compliant_files

# Specify the directory to check
directory = '/Users/astrid/Documents/Thesis/MEOWS/3everythinglabelledPaddedRenamed'

# Call the function and print the non-compliant files
non_compliant_files = check_filenames(directory)
print("Files without proper identifier:", non_compliant_files)
