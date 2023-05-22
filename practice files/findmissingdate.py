import os
from datetime import datetime, timedelta

# Function to parse the date from the filename
def parse_date(filename):
    # Remove the '.zip' extension from the filename
    filename = os.path.splitext(filename)[0]
    parts = filename.split('-')
    if len(parts) != 5:
        return None
    try:
        date = datetime.strptime('-'.join(parts[2:5]), '%Y-%m-%d')
        return date
    except ValueError:
        return None

# Start and end dates
start_date = datetime(2021, 3, 1)
end_date = datetime(2023, 5, 15)

# Set of all dates that should exist
all_dates = {start_date + timedelta(days=i) for i in range((end_date-start_date).days + 1)}

# Directory where the files are located
directory = r"C:\Github\PricePrediction\csvfiles\BTCUSDT"

# Set of existing dates
existing_dates = set()

# Iterate over all the files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".csv"):  # If the file is a csv file
        date = parse_date(filename)
        if date is not None:
            existing_dates.add(date)

# Find the missing dates
missing_dates = all_dates - existing_dates

# Print the missing dates
for datecheck in sorted(missing_dates):
    print(datecheck.strftime('%Y-%m-%d'))


