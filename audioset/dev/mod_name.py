import os

target_dir = '/Users/astrid/Documents/Thesis/MEOWS/catmeow-resampled'

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

for filename in os.listdir(target_dir):
    # Split the filename to extract the target class
    parts = filename.split('_')
    target_class = parts[0]  # Convert to integer for comparison
    # print(target_class)

    # Determine the new class name based on the target class
    if target_class in ['1', '2', '4']:
        new_class_name = "YOUNG_ADULT"
    elif target_class in ['5', '6', '7']:
        new_class_name = "ADULT"
    elif target_class in ['9', '13']:
        print("OK")
        new_class_name = "SENIOR"
    else:
        continue  # Skip renaming if the class does not match any criteria

    # Construct the new filename
    new_filename = '_'.join([new_class_name] + parts[1:])

    # Construct the full old and new file paths
    o_fld_file = os.path.join(target_dir, filename)
    newile = os.path.join(target_dir, new_filename)

    # Rename the file
    os.rename(old_file, new_file)



# for filename in os.listdir(target_dir):
#     if 'WHO01' in filename:
#         # Construct the new filename
#         new_filename = '2_' + filename
#         # Construct the full old and new file paths
#         old_file = os.path.join(target_dir, filename)
#         new_file = os.path.join(target_dir, new_filename)
#         # Rename the file
#         os.rename(old_file, new_file)