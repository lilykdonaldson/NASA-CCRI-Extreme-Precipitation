import pandas as pd
import matplotlib.pyplot as plt
import os
import calendar

# Path to the folder containing the CSV files
folder_path = '/Users/lilydonaldson/Downloads/examples/current_scripts/NYC_IMERG_Data'

# Initialize an empty DataFrame to store all data
all_data = pd.DataFrame()

# Loop through each year from 2001 to 2023
for year in range(2001, 2023):
    file_path = os.path.join(folder_path, f'{year}.csv')
    # Read the CSV file, skipping the first 9 rows (metadata)
    yearly_data = pd.read_csv(file_path, skiprows=9)
    # Assume the first column is the timestamp and the second is precipitation
    yearly_data.columns = ['Timestamp', 'Precipitation']
    # Convert Timestamp to datetime
    yearly_data['Timestamp'] = pd.to_datetime(yearly_data['Timestamp'])
    # Append to the main DataFrame
    all_data = all_data.append(yearly_data)

# Function to annotate percentiles on the plot
def annotate_percentiles(ax, data, color='red'):
    for percentile in [95, 99]:
        value = data.quantile(percentile / 100)
        ax.axvline(value, color=color, linestyle='dashed', linewidth=1)
        ax.text(value, 0.3, f'{percentile}th percentile: {value:.2f}', rotation=45, color=color)

# Directory for saving plots
plots_dir = '/Users/lilydonaldson/Downloads/examples/current_scripts/NYC_IMERG_Plots'
os.makedirs(plots_dir, exist_ok=True)

# Histogram for the combined data (2001-2022)
plt.figure(figsize=(10, 6))
plt.hist(all_data['Precipitation'], bins=50, log=True, density=True)
plt.title('NYC Precipitation Histogram (2001-2022)')
plt.xlabel('Precipitation (mm/hour)')
plt.ylabel('Density')
plt.xlim(0, 11)
plt.ylim(10**-4, 10**1)
ax = plt.gca()
annotate_percentiles(ax, all_data['Precipitation'])
plt.savefig(os.path.join(plots_dir, 'NYC_Precipitation_Histogram_2001_2022.png'))
plt.close()

# Histograms for each month across all years
for month in range(1, 13):
    monthly_data = all_data[all_data['Timestamp'].dt.month == month]
    plt.figure(figsize=(10, 6))
    plt.hist(monthly_data['Precipitation'], bins=50, log=True, density=True)
    # Use calendar module to get month name
    month_name = calendar.month_name[month]
    plt.title(f'NYC Precipitation Histogram for {month_name} (2001-2022)')
    plt.xlabel('Precipitation (mm/hour)')
    plt.ylabel('Density')
    plt.xlim(0, 11)
    plt.ylim(10**-4, 10**1)
    ax = plt.gca()
    annotate_percentiles(ax, monthly_data['Precipitation'])
    plt.savefig(os.path.join(plots_dir, f'NYC_Precipitation_Histogram_{month_name}_2001_2022.png'))
    plt.close()
