import os
import shutil

# Path to your main folder that has all 600 images
source_folder = r'C:\Users\admin\Desktop\Python_jupyter\ML_LEARN\DL college\data\Cat-Dog_Pandas-college\Test'

# Destination folders for each class
cat_folder = os.path.join(source_folder, "cats")
dog_folder = os.path.join(source_folder, "dogs")
panda_folder = os.path.join(source_folder, "pandas")

# Create destination folders if they don't exist
os.makedirs(cat_folder, exist_ok=True)
os.makedirs(dog_folder, exist_ok=True)
os.makedirs(panda_folder, exist_ok=True)

# Get list of all image files sorted by name (to maintain order)
files = sorted(os.listdir(source_folder))

# Split based on index range
for i, file in enumerate(files):
    file_path = os.path.join(source_folder, file)
    
    # Skip folders that were just created
    if os.path.isdir(file_path):
        continue

    if i < 200:
        shutil.move(file_path, os.path.join(cat_folder, file))
    elif i < 400:
        shutil.move(file_path, os.path.join(dog_folder, file))
    else:
        shutil.move(file_path, os.path.join(panda_folder, file))

print("Images successfully split into cats, dogs, and pandas folders.")
