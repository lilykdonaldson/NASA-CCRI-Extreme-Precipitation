'''
Save Modified IMERG Files
--part of the IMERG-GISS-comparison script package--
Description: This script takes raw subdaily IMERG netCDF files and outputs averaged 
monthly netCDF files and .npz saved arrays of the chosen variable, including by region, 
with an option to regrid and resample the data. The .npz saved arrays are 3D numpy arrays 
with dimensions like (num_of_datapoints, lat, lon) such as (120, 90, 144) where 120 is the 
number of datapoints (30 days at a 6 hour sampling rate) and 90, 144 which are the dimensions
of the grid. The .npz saved arrays can be used with IMERG_GISS_hist_stats.py included in this script package.

Lily Donaldson [agency]<lily.k.donaldson@nasa.gov> [evergreen]<lilykdonaldson@gmail.com>
January 2024, Developed with Python 3.9.13

This script takes the following user inputs which are set in the "USER INPUTS" section:
	-- start_year and end_year: the first and last year in your chosen time period of data to process
	-- original_data_folder: pathname to folder that contains the original IMERG data which should be 
	   organized in year folders. Every file in each year folder that ends with '.nc4' will be opened. 
	   This code was tested with data from 'GPM IMERG Final Precipitation L3 Half Hourly 0.1 degree x 0.1 
	   degree V07' which can be downloaded (open access) from NASA GES DISC.
	-- mask_folder: the folder where the region masks are stored. The region masks should be netCDF 
	   files with variables 'latitude', 'longitude', and 'mask' where the mask variable should contain 
	   '1' within the region mask boundaries and '0' outside of the region mask boundaries. 
	-- regions: a list of regions that correspond to the chosen region masks. Where a region is named
	   like 'contUSA', its corresponding region mask within the mask folder is assumed to be named like
	   'contUSA_mask.nc'. 
	-- variable_name: the name of the variable of interest. For IMERG data, this will likely be 
	   "precipitation". The new netCDF datasets will only contain the variable of interest (averaged 
	   across the time period, such as by month), latitude, and longitude.
	-- regrid and regrid_file: regrid should be set to True if you would like the data to be regridded.
	   If regrid is set to True, regrid_file should be the path to a netCDF file which contains lat and
	   lon variables with the correct grid. The script 'createRegridFile.py' shows an example of how to
	   create this file. If you do not wish to regrid the data, set regrid to False, and you will not
	   need to change regrid_file from the example.
	-- resample and resample_rate: resample should be set to True if you would like the data to be re-
	   sampled. If resample is set to True, resample_rate should be the desired sampling rate in hours.
	   For example, if resample_rate is set to 6, the data will be resampled to one measurement every
	   six hours. If you do not wish to resample the data, set resample to False, and you will not
	   need to change resample_rate from the example.
	-- unit_conversion_factor: The factor by which to convert the variable of interest. Set this to 1 if
	   no unit conversion is needed. IMERG precipitation has units mm/hour; use 24 as a factor to convert 
	   precipitation to mm/day.
	-- output_folder: the folder where the generated files will be saved.

Example File Organization
	-current directory
	-?-original_data_folder
	----2010
	-----3B-HHR.MS.MRG.3IMERG.20100101-S000000-E002959.0000.V07A.HDF5.nc4
	----2011
	-----3B-HHR.MS.MRG.3IMERG.20110101-S000000-E002959.0000.V07A.HDF5.nc4
	----2012
	-----3B-HHR.MS.MRG.3IMERG.20120101-S000000-E002959.0000.V07A.HDF5.nc4
	-?-output_folder
	----2011
	-----region1
	------2011_09_region1_precipitation_average.nc
	------2011_09_region1_precipitation.npz
	-----region2
	------2011_09_region2_precipitation_average.nc
	------2011_09_region2_precipitation.npz
'''
#TO_DOs: script runs quite slowly. implement optimization (code review region looping and multiprocessing)
#month/season/year in another script where arrays are simply added for those modes
#---------------------------IMPORTS--------------------------------#
import os
import re
import warnings
from typing import List
from datetime import datetime, timedelta
import calendar
import xarray as xr #developed with v.0.20.1
import pandas as pd #developed with v.1.4.4
import numpy as np #developed with v.1.24.3
import cftime #developed with v.1.6.3

warnings.filterwarnings("ignore", message="invalid value encountered in cast")

#------------------------------------------------------------------#

#---------------------------USER INPUTS--------------------------------#
start_year = 2010
end_year = 2020
original_data_folder = "/Users/lilydonaldson/Downloads/examples/data/IMERG/IMERG_subdaily_raw"
mask_folder = "/Users/lilydonaldson/Downloads/examples/masks"
regions = [
	#'northeast'
	#'nyc'
	# 'southwest', 'southeast', 'southerngreatplains', 'midwest',
	# 'northerngreatplains', 'northeast', 'northwest', 'contUSA'
]
variable_name = "precipitation"
regrid = True
regrid_file = "/Users/lilydonaldson/Downloads/examples/regrid_files/regrid_2x2-5.nc" #regridding is only performed if regrid is set to True
resample = True
resample_rate = 6 #resampling is only performed if resample is set to True.
unit_conversion_factor = 24 #set this to 1 if no conversion is needed. 
output_folder = "/Users/lilydonaldson/Downloads/examples/data/IMERG/IMERG_regrid/lastpass_regridded"
#-------------------------END OF USER INPUTS----------------------------#


#---------------------------FUNCTIONS--------------------------------#
def process_nc_files(years: list, input_folder_path_base: str, output_folder_path_base: str, 
	chosen_variable: str, regrid: bool, resample: bool, regions: list, mask_folder: str, regrid_file: str = None, 
	resample_rate: int = None, unit_conversion_factor: float = 1.0):
	"""
	Processes .nc4 files by regridding, resampling, extracting a chosen variable, and combining them by month
	to generate two intermediate files per region. The files generated are a netCDF file which contains data 
	for 1 month with the chosen variable averaged and an .npz compressed numpy file which contains a flattened 
	array of all of the chosen variable's values.
	:param years: List of years to process.
	:param input_folder_path_base: Base path to the folder containing .nc4 files.
	:param output_folder_path_base: Base path to the folder for saving output files.
	:param chosen_variable: The variable to be extracted from the files.
	:param regrid: Boolean indicating whether to regrid the data.
	:param resample: Boolean indicating whether to resample the data.
	:param regions: a list of region names.
	:param mask_folder: a path name to a folder which contains .nc mask files corresponding to each of the regions.
	:param regrid_file: Path to the file containing the new grid for regridding.
	:param resample_rate: Rate at which to resample the data (in hours).
	:param unit_conversion_factor: Factor to multiply the variable by for unit conversion.
	"""

	def extract_year_month(filename):
		match = re.search(r"\d{8}", filename)
		if match:
			date = pd.to_datetime(match.group(), format='%Y%m%d')
			return date.year, date.month
		return None
	if not os.path.exists(output_folder_path_base):
		os.makedirs(output_folder_path_base)

	for year in years:
		input_folder_path = os.path.join(input_folder_path_base, str(year))
		output_folder_path = os.path.join(output_folder_path_base, str(year))
		if not os.path.exists(output_folder_path):
			os.makedirs(output_folder_path)
		# if 'global' not in regions:
		# 	regions.append('global')
		if regrid and regrid_file:
			# Load the regrid file to get the new grid
			regrid_dataset = xr.open_dataset(regrid_file)
		files_by_month = {}
		for filename in os.listdir(input_folder_path):
			if filename.endswith('.nc4'):
				year_month = extract_year_month(filename)
				if year_month:
					files_by_month.setdefault(year_month, []).append(filename)
		for region in regions:
			# Create a folder for each region inside the base output directory
			output_directory = os.path.join(output_folder_path, region)
			if not os.path.exists(output_directory):
				os.makedirs(output_directory)
			for (year, month), files in files_by_month.items():
				all_values = []
				monthly_datasets = []
				for file in files:
					file_path = os.path.join(input_folder_path, file)
					with xr.open_dataset(file_path) as ds:
						if resample and resample_rate:
							# Extract the time variable from the dataset
							time_var = ds['time'].values[0]
							# Use cftime to convert the time variable
							if isinstance(time_var, cftime.datetime):
								timestamp = cftime.datetime(time_var.year, time_var.month, time_var.day, time_var.hour, time_var.minute)
							else:
								timestamp = pd.to_datetime(time_var)
							# Check if the timestamp is at the desired resampling interval
							if timestamp.hour % resample_rate != 0 or timestamp.minute != 0:
								continue  # Skip this file
						if regrid_file:
							with warnings.catch_warnings():
								warnings.simplefilter("ignore", FutureWarning)
								ds = ds.interp(
									lat=regrid_dataset['lat'],
									lon=regrid_dataset['lon'],
									method='nearest',
									kwargs={'fill_value': None}
								)

						if region != 'global':
							mask_file = f'{mask_folder}/{region}_mask.nc'
							mask = xr.open_dataset(mask_file)
							mask = mask.rename({'latitude': 'lat', 'longitude': 'lon'})
							mask = mask.reindex(lat=np.append(np.insert(mask.lat.values, 0, -90), 90), fill_value=0)
							ds[chosen_variable] = ds[chosen_variable].where(mask['mask'])

						variable_data = ds[chosen_variable]
						if unit_conversion_factor != 1.0:
							# Apply unit conversion if unit_conversion_factor is not 1.0
							variable_data *= unit_conversion_factor
						all_values.append(variable_data.values.flatten())
						monthly_datasets.append(variable_data)

				# Combine all values for the month into a single array and save as .npz
				all_values_combined = np.concatenate(all_values)
				npz_output_filename = f"{year}_{month:02d}_{region}_{chosen_variable}.npz"
				npz_output_path = os.path.join(output_directory, npz_output_filename)
				np.savez_compressed(npz_output_path, all_values=all_values_combined)
				# Calculate the average across the month and save as .nc
				average_data = xr.concat(monthly_datasets, dim='time').mean(dim='time')
				nc_output_filename = f"{year}_{month:02d}_{region}_{chosen_variable}_average.nc"
				nc_output_path = os.path.join(output_directory, nc_output_filename)
				average_data.to_netcdf(nc_output_path)
				print(f'Completed processing files from {calendar.month_name[month]} {year} for {region} region.')
#----------------------------------END OF FUNCTIONS--------------------------------#

#-------------------------------MAIN CODE-----------------------------------#
if __name__ == "__main__":
	years = list(range(start_year, end_year+1))
	process_nc_files(years, original_data_folder, output_folder, variable_name, regrid, resample, regions, mask_folder, regrid_file, resample_rate, unit_conversion_factor) 
#---------------------------------END OF MAIN CODE---------------------------------#

