'''
Save Modified GISS Files
--part of the IMERG-GISS-comparison script package--
Description: This script takes raw subdaily GISS netCDF files and outputs averaged 
monthly netCDF files and .npz saved arrays of the chosen variable, including by region. 
The .npz saved arrays are 3D numpy arrays with dimensions like (num_of_datapoints, lat, lon) 
such as (120, 90, 144) where 120 is the number of datapoints (30 days at a 6 hour sampling rate) 
and 90, 144 which are the dimensions of the grid. The .npz saved arrays can be used with 
IMERG_GISS_hist_stats.py included in this script package.
 

Lily Donaldson [agency]<lily.k.donaldson@nasa.gov> [evergreen]<lilykdonaldson@gmail.com>
January 2024, Developed with Python 3.9.13

This script takes the following user inputs which are set in the "USER INPUTS" section:
	-- start_year and end_year: the first and last year in your chosen time period of data to process
	-- original_data_folder: pathname to folder that contains the original subdaily GISS data which should be 
	   organized in year folders. The netCDF files must be named like "DEC2019.aijh12iWISO_20th_MERRA2_ANL.nc" 
	   or line 94 (file_name) must be updated.
	-- mask_folder: the folder where the region masks are stored. The region masks should be netCDF 
	   files with variables 'latitude', 'longitude', and 'mask' where the mask variable should contain 
	   '1' within the region mask boundaries and '0' outside of the region mask boundaries. 
	-- regions: a list of regions that correspond to the chosen region masks. Where a region is named
	   like 'contUSA', its corresponding region mask within the mask folder is assumed to be named like
	   'contUSA_mask.nc'. 
	-- variable_name: the name of the variable of interest. For GISS data, this will likely be 
	   "prec". The new netCDF datasets will only contain the variable of interest (averaged 
	   across the month), latitude, and longitude.
	-- output_folder: the folder where the generated files will be saved.


Example File Organization
	-current directory
	-?-original_data_folder
	----2020
	-----JAN2020.aijh12iWISO_20th_MERRA2_ANL.nc
	----2019
	-----JAN2019.aijh12iWISO_20th_MERRA2_ANL.nc
	-?-output_folder
	----2020
	-----region1
	------2020_01_region1_prec_average.nc
	------2020_01_region1_prec.npz
	-----region2
	------2020_01_region2_prec_average.nc
	------2020_01_region2_prec.npz
'''

#---------------------------IMPORTS--------------------------------#
import os
import xarray as xr #developed with v.0.20.1
import numpy as np #developed with v.1.24.3

#------------------------------------------------------------------#

#---------------------------USER INPUTS--------------------------------#
start_year = 2012
end_year = 2022
original_data_folder = "/Users/lilydonaldson/Downloads/examples/data/GISS/GISS_subdaily"
mask_folder = "/Users/lilydonaldson/Downloads/examples/masks"
regions = [
	'northeast'
	#'nyc'
	# 'southwest', 'southeast', 'southerngreatplains', 'midwest',
	# 'northerngreatplains', 'northeast', 'northwest', 'contUSA'
]
variable_name = "prec"
output_folder = "/Users/lilydonaldson/Downloads/examples/data/GISS/GISS_automated/northeast_nearest_automated_GISS"
#-------------------------END OF USER INPUTS----------------------------#

#---------------------------FUNCTIONS--------------------------------#

def process_nc_files(years: list, input_folder_path_base: str, output_folder_path_base: str, 
	chosen_variable: str, regions: list, mask_folder: str):
	"""
	Processes GISS .nc files by extracting a chosen variable to generate two intermediate files per month 
	of each year and for every region. The files generated are a netCDF file which contains data for 1 
	month with the chosen variable averaged) and an .npz compressed numpy file which contains a flattened 
	array of all of the chosen variable's values for that month.
	:param years: List of years to process.
	:param input_folder_path_base: Base path to the folder containing original .nc files.
	:param output_folder_path_base: Base path to the folder for saving output files.
	:param chosen_variable: The variable to be extracted from the files.
	:param regions: a list of region names.
	:param mask_folder: a path name to a folder which contains .nc mask files corresponding to each of the regions.
	"""

	month_names = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
	if(chosen_variable=='prec'):
		new_variable_name = 'precipitation'
	else:
		new_variable_name = chosen_variable
	if not os.path.exists(output_folder_path_base):
		os.makedirs(output_folder_path_base)
	for year in years:
		input_folder_path = os.path.join(input_folder_path_base, str(year))
		output_folder_path = os.path.join(output_folder_path_base, str(year))
		if not os.path.exists(output_folder_path):
			os.makedirs(output_folder_path)
		# if 'global' not in regions:
		# 	regions.append('global')
		for region in regions:
			# Create a folder for each region inside the base output directory
			output_directory = os.path.join(output_folder_path, region)
			if not os.path.exists(output_directory):
				os.makedirs(output_directory)

			if region!='global':
				mask_file = f'{mask_folder}/{region}_mask.nc'
				mask = xr.open_dataset(mask_file)
				mask = mask.rename({'latitude': 'lat', 'longitude': 'lon'})
				mask = mask.reindex(lat=np.append(np.insert(mask.lat.values, 0, -90), 90), fill_value=0)

			all_values = []
			monthly_datasets = []
			for month in month_names:
				month_number = month_names.index(month) + 1
				file_name = f'{month}{year}.aijh12iWISO_20th_MERRA2_ANL.nc'
				file_path = os.path.join(input_folder_path, file_name)
				# Check if the file exists
				if os.path.exists(file_path):
					# Load the dataset for the current month
					dataset = xr.open_dataset(file_path)
					if region!='global':
						# Apply the mask to the dataset
						dataset[chosen_variable] = dataset[chosen_variable].where(mask['mask'])

					lat = dataset['lat']
					lon = dataset['lon']
					prec = dataset[chosen_variable]
					prec_averaged = np.mean(prec, axis=0)
					averageddataset = xr.Dataset(
					    data_vars={new_variable_name: (['lat', 'lon'], prec_averaged.data)}, 
					    coords={'lat': lat, 'lon': lon}  # Define 'lat' and 'lon' as coordinates
					)
					# Save the masked dataset to a new netCDF file in the region-specific folder
					output_file = f"{year}_{month_number:02d}_{region}_{new_variable_name}_average.nc"
					nc_output_path = os.path.join(output_directory, output_file)
					averageddataset.to_netcdf(nc_output_path)

					variable_data = dataset[chosen_variable].values
					npz_output_filename = f"{year}_{month_number:02d}_{region}_{new_variable_name}.npz"
					npz_output_path = os.path.join(output_directory, npz_output_filename)
					np.savez_compressed(npz_output_path, all_values=variable_data)
					print(f'Completed processing files from {month} {year} for {region} region.')

#----------------------------------END OF FUNCTIONS--------------------------------#


#-------------------------------MAIN CODE-----------------------------------#
if __name__ == "__main__":
	years = list(range(start_year, end_year+1))
	process_nc_files(years, original_data_folder, output_folder, variable_name, regions, mask_folder) 

#---------------------------------END OF MAIN CODE---------------------------------#


