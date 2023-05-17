import zipfile
import os

# Directory where the zip files are located
directory = r"C:\Github\PricePrediction\btc_data"

# Iterate over all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".zip"):  # If the file is a ZIP file
        zip_path = os.path.join(directory, filename)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(directory)
        # Remove the zip file after extraction
        os.remove(zip_path)
