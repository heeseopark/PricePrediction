import os
import requests
import zipfile
from io import BytesIO
from bs4 import BeautifulSoup

# URL of the page
url = 'https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1m/'

# Directory to save the CSV files
save_dir = 'C:\Github\PricePrediction\\btc_data'

# Make the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Get the page content
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Loop over each link
for link in soup.find_all('a'):
    href = link.get('href')
    
    # Only process if it's a zip file
    if href.endswith('.zip'):
        print(f'Downloading and extracting {href}...')
        
        # Download the zip file
        r = requests.get(href)
        z = zipfile.ZipFile(BytesIO(r.content))
        
        # Extract the CSV file
        for file in z.namelist():
            if file.endswith('.csv'):
                # Define the save path
                save_path = os.path.join(save_dir, os.path.basename(file))
                
                # Extract the file
                with z.open(file) as f:
                    content = f.read()
                
                # Write the file
                with open(save_path, 'wb') as f:
                    f.write(content)
                print(f'{file} has been downloaded')
