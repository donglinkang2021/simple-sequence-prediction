import pandas as pd
from datetime import datetime

# Read the Excel file
file_path = 'data/4single.xlsx'
df = pd.read_excel(file_path)

# Convert 'Date & Time' to datetime
df['Date & Time'] = pd.to_datetime(df['Date & Time'], format='%d-%m-%Y %H:%M:%S')

# Calculate the time difference in hours from the first timestamp
df['time(h)'] = (df['Date & Time'] - df['Date & Time'].iloc[0]).dt.total_seconds() / 3600

# Convert formaldehyde from ppm to µg/m³ (assuming 1 ppm = 1240 µg/m³ for formaldehyde)
df['formaldehyde(ug/m³)'] = df['A: Formaldehyde(ppm)'] * 1240

# Select the required columns
df = df[['time(h)', 'formaldehyde(ug/m³)']]

# Save to CSV
output_file_path = 'data/4single.csv'
df.to_csv(output_file_path, index=False)

# Print the head of the dataframe
print(df.head())
