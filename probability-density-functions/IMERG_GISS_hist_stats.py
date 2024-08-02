'''
IMERG VS GISS COMPARISON
--part of the IMERG-GISS-comparison script package--
Description: This script takes intermediate files generated by saveIMERGfiles.py and saveGISSfiles.py 
and generates visualizations and stats comparing IMERG and GISS data.

Lily Donaldson [agency]<lily.k.donaldson@nasa.gov> [evergreen]<lilykdonaldson@gmail.com>
January 2024, Developed with Python 3.9.13

This script takes the following user inputs which are set in the "USER INPUTS" section:
	-- start_year and end_year: the first and last year in your chosen time period of data to process
	-- GISS_data_folder: pathname to folder that contains the intermediate GISS data files generated by
	   saveGISSfiles.py
	-- IMERG_data_folder: pathname to folder that contains the intermediate GISS data files generated by
	   saveIMERGfiles.py
	-- variable_name: the name of the variable of interest. This will likely be 'precipitation'.
	-- output_folder: the folder where the comparison vizualizations will be saved.
	-- regions: a list of regions that you would like the vizualizations to be performed on. Each region
	   should have its own intermediate files generated by the aforementioned scripts.
	-- mode: the mode the data should be analyzed with which can be 'month','years', 'single-year', or 'season'.
			- month mode generates plots for each month listed in months_list for the chosen data. For 
			  example, if start_year is 2019, end_year is 2020, mode is 'month', and months_list is ['JAN',
			  'DEC'] then visualizations will be created for January using 2019 and 2020 data and for
			  December using 2019-2020 data. If the months_list contains "ALL" as a list item, all 12 months
			  will be processed. 
			 - season mode generates plots in the same manner as month mode but for a chosen season (which
			   must be set in the chosen_season variable)
			 - years mode generates visualizations for sequences of years. For example, if mode is set to 'years', 
			   start_year is 2019, and end_year is 2020, visualizations will be created for all data from
			   2019-2020.
			 - single-year mode generates visualizations for individual years. For example, if mode is set 
			   to 'single-year', start_year is 2019, and end_year is 2020, visualizations will be created for 
			   2019 and 2020 separately. 
	-- months_list: a list of months to process when mode is set to 'month' or in custom season mode.
	-- chosen_season: when mode = 'season', this is the season for which to draw data from, which can be 
	   'winter' (DEC-JAN-FEB), 'spring' (MAR-APR-MAY), 'summer' (JUN-JUL-AUG), 'fall' (SEP-OCT-NOV), 'all' 
	   (creates visualizations for all 4 of the pre-programmed seasons separately), or 'custom' which uses 
	   the months listed in months_list to create a custom season. For example, if start_year is set to 2019, 
	   end_year is set to 2020, mode is set to 'season' and chosen_season is set to 'spring', visualizations 
	   will be created for spring (MAR-APR-MAY) using data from 2019 and 2020. If chosen_season is instead set 
	   to 'all', visualizations will be created for winter, spring, summer, and fall using data from 2019 and 
	   2020. If chosen_season is instead set to 'custom' and months_list is set to ['FEB','MAR','APR'], 
	   visualizations will be created for February, March, and April collectively using data from 2019 and
	   2020.


Example File Organization
	-current directory
	-?-GISS_data_folder
	----2020
	-----region1
	------2020_01_region1_precipitation_average.nc
	------2020_01_region1_precipitation.npz
	-?-IMERG_data_folder
	----2020
	-----region1
	------2020_01_region1_precipitation_average.nc
	------2020_01_region1_precipitation.npz
	-?-output_folder

'''

#---------------------------IMPORTS--------------------------------#
import os
import xarray as xr #developed with v.0.20.1
import numpy as np #developed with v.1.24.3
import matplotlib.pyplot as plt #developed with v.3.4.1

#------------------------------------------------------------------#

#---------------------------USER INPUTS--------------------------------#
start_year = 2012
end_year = 2022
GISS_data_folder = "/Users/lilydonaldson/Downloads/examples/data/GISS/GISS_automated/NYC_automated_GISS"
IMERG_data_folder = "/Users/lilydonaldson/Downloads/examples/data/IMERG/IMERG_automated/NYC_automated"
variable_name = "precipitation"
output_folder = "/Users/lilydonaldson/Downloads/" #examples/visualizations/GISSIMERGcompare"
regions = [ 'nyc']
# 	'southwest','southeast','global',
# 	'southerngreatplains', 'midwest', 'global',
# 	'northerngreatplains', 'northeast', 'northwest', 'contUSA'
# ]
mode = 'season'

chosen_season = 'all' #this only needs to be changed if mode='season'
months_list = ['ALL'] #this only needs to be changed if mode='month' or mode='season' and chosen_season = 'custom'

#-------------------------END OF USER INPUTS----------------------------#

#---------------------------FUNCTIONS--------------------------------#

def createCompareViz(mode: str, years: list, regions: list, chosen_variable: str, GISS_data_folder: str, IMERG_data_folder: str, 
	output_folder_path_base: str, chosen_season: str = '', months_list: list = [None]):
	"""
	Creates histograms and statistical tables to compare GISS and IMERG data.
	:param mode: The data visualization mode which can be month, year, single_year, or season.
	:param years: List of years to process.
	:param regions: a list of region names.
	:param chosen_variable: The name of the variable the intermediate files contain. 
	:param GISS_data_folder: Base path to the folder containing the GISS intermediate files.
	:param IMERG_data_folder: Base path to the folder containing the GISS intermediate files.
	:param output_folder_path_base: Base path to the folder for saving output files.
	:param chosen_season: The season to analyze when in season mode.
	:param months_list: The list of months to analyze if in month mode or season mode with a custom season.
	"""
	month_dict = {
	    'JAN': '01',
	    'FEB': '02',
	    'MAR': '03',
	    'APR': '04',
	    'MAY': '05',
	    'JUN': '06',
	    'JUL': '07',
	    'AUG': '08',
	    'SEP': '09',
	    'OCT': '10',
	    'NOV': '11',
	    'DEC': '12'
	}
	region_dict = {
	    'global': 'Global',
	    'southwest': 'Southwest USA',
	    'southeast': 'Southeast USA',
	    'midwest': 'Midwest USA',
	    'northeast': 'Northeast USA',
	    'northwest': 'Northwest USA',
	    'northerngreatplains': 'Northern Great Plains USA',
	    'southerngreatplains': 'Southern Great Plains USA',
	    'contUSA': 'Contiguous USA',
	    'nyc': 'New York City',
	    'northeastcoast': 'Northeast USA Coast'
	}
	month_strings = [f'{i:02d}' for i in range(1, 13)]
	if mode=='single-year':
		#histograms = []
		for year in years:
			for region in regions:
				giss_values_list = []
				imerg_values_list = []
				base_folder = f"{year}/{region}/"
				for month in month_strings:
					file_name = f'{year}_{month}_{region}_{chosen_variable}.npz'
					file_path = os.path.join(GISS_data_folder, base_folder, file_name)
					data = np.load(file_path)
					giss_values_list.append(data['all_values'].flatten())
					file_path = os.path.join(IMERG_data_folder, base_folder, file_name)
					data = np.load(file_path)
					imerg_values_list.append(data['all_values'].flatten())
				imerg_values_array = np.concatenate(imerg_values_list)
				giss_values_array = np.concatenate(giss_values_list)
				imerg_values_array = imerg_values_array[~np.isnan(imerg_values_array)]
				giss_values_array = giss_values_array[~np.isnan(giss_values_array)]
				region_name = region_dict.get(region)
				if region_name is None:
				    region_name = region
				title = f"{year} {region_name}"
				print(f"--- finished data concatenation for {year}, {region}.")
				save_file = os.path.join(output_folder_path_base, f"{year}_{region}_singleyear_histogram.png")
				histogramGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
				save_file = os.path.join(output_folder_path_base, f"{year}_{region}_singleyear_table.png")
				statsTableGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
				#histograms.append(save_file)
	elif mode=='years':
		for region in regions:
			giss_values_list = []
			imerg_values_list = []
			for year in years:
				base_folder = f"{year}/{region}/"
				for month in month_strings:
					file_name = f'{year}_{month}_{region}_{chosen_variable}.npz'
					file_path = os.path.join(GISS_data_folder, base_folder, file_name)
					data = np.load(file_path)
					giss_values_list.append(data['all_values'].flatten())
					file_path = os.path.join(IMERG_data_folder, base_folder, file_name)
					data = np.load(file_path)
					imerg_values_list.append(data['all_values'].flatten())
			imerg_values_array = np.concatenate(imerg_values_list)
			giss_values_array = np.concatenate(giss_values_list)
			imerg_values_array = imerg_values_array[~np.isnan(imerg_values_array)]
			giss_values_array = giss_values_array[~np.isnan(giss_values_array)]
			region_name = region_dict.get(region)
			if region_name is None:
				region_name = region
			title = f"{years[0]}-{years[-1]} {region_name}"
			print(f"--- finished data concatenation for {years}, {region}.")
			print("GISS: ",len(giss_values_array))
			print("IMERG: ",len(imerg_values_array))
			save_file = os.path.join(output_folder_path_base, f"{years[0]}-{years[-1]}_{region}_severalyears_histogram.png")
			histogramGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
			save_file = os.path.join(output_folder_path_base, f"{years[0]}-{years[-1]}_{region}_severalyears_table.png")
			statsTableGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
	elif mode=='month':
		for region in regions:
			if "ALL" in months_list:
				months_list = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
			for month in months_list:
				giss_values_list = []
				imerg_values_list = []
				for year in years:
						base_folder = f"{year}/{region}/"
						file_name = f'{year}_{month_dict.get(month)}_{region}_{chosen_variable}.npz'
						file_path = os.path.join(GISS_data_folder, base_folder, file_name)
						data = np.load(file_path)
						giss_values_list.append(data['all_values'].flatten())
						file_path = os.path.join(IMERG_data_folder, base_folder, file_name)
						data = np.load(file_path)
						imerg_values_list.append(data['all_values'].flatten())
				imerg_values_array = np.concatenate(imerg_values_list)
				giss_values_array = np.concatenate(giss_values_list)
				imerg_values_array = imerg_values_array[~np.isnan(imerg_values_array)]
				giss_values_array = giss_values_array[~np.isnan(giss_values_array)]
				region_name = region_dict.get(region)
				if region_name is None:
					region_name = region
				title = f"{month}, {years[0]}-{years[-1]} {region_name}"
				print(f"--- finished data concatenation for {month}, {region}.")
				save_file = os.path.join(output_folder_path_base, f"{month}_{years[0]}-{years[-1]}_{region}_histogram.png")
				histogramGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
				save_file = os.path.join(output_folder_path_base, f"{month}_{years[0]}-{years[-1]}_{region}_table.png")
				statsTableGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
			
	elif mode =='season':
		def season_mode():
			for region in regions:
				giss_values_list = []
				imerg_values_list = []
				for month in months_list:
					for year in years:
						base_folder = f"{year}/{region}/"
						file_name = f'{year}_{month_dict.get(month)}_{region}_{chosen_variable}.npz'
						file_path = os.path.join(GISS_data_folder, base_folder, file_name)
						data = np.load(file_path)
						giss_values_list.append(data['all_values'].flatten())
						file_path = os.path.join(IMERG_data_folder, base_folder, file_name)
						data = np.load(file_path)
						imerg_values_list.append(data['all_values'].flatten())
				imerg_values_array = np.concatenate(imerg_values_list)
				giss_values_array = np.concatenate(giss_values_list)
				imerg_values_array = imerg_values_array[~np.isnan(imerg_values_array)]
				giss_values_array = giss_values_array[~np.isnan(giss_values_array)]
				region_name = region_dict.get(region)
				if region_name is None:
					region_name = region
				title = f"{chosen_season}, {years[0]}-{years[-1]} {region_name}"
				print(f"--- finished data concatenation for {chosen_season}, {region}.")
				save_file = os.path.join(output_folder_path_base, f"{chosen_season}_{years[0]}-{years[-1]}_{region}_histogram.png")
				histogramGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
				save_file = os.path.join(output_folder_path_base, f"{chosen_season}_{years[0]}-{years[-1]}_{region}_table.png")
				statsTableGISSIMERG(giss_values_array,imerg_values_array,title,save_file)
		if chosen_season == "winter":
			months_list = ['DEC','JAN','FEB']
		elif chosen_season == "spring":
			months_list = ['MAR','APR','MAY']
		elif chosen_season == "summer":
			months_list = ['JUN','JUL','AUG']
		elif chosen_season == "fall":
			months_list = ['SEP','OCT','NOV']
		elif chosen_season == "custom":
			for month in months_list:
				if month not in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']:
					print("an invalid month was selected in the months_list variable.")
					return None
		elif chosen_season == "all":
			pass
		else:
			print("invalid chosen_season. choose winter, spring, summer, fall, or custom.")
			return None
		if chosen_season!='all':
			season_mode()
		else:
			chosen_season = "winter"
			months_list = ['DEC','JAN','FEB']
			season_mode()
			chosen_season = "spring"
			months_list = ['MAR','APR','MAY']
			season_mode()
			chosen_season = "summer"
			months_list = ['JUN','JUL','AUG']
			season_mode()
			chosen_season = "fall"
			months_list = ['SEP','OCT','NOV']
			season_mode()
	else:
		print("invalid mode")


def histogramGISSIMERG(giss_data, imerg_data, title, output_path):
    # Calculate statistics
    giss_avg, giss_95th, giss_99th = np.mean(giss_data), np.percentile(giss_data, 95), np.percentile(giss_data, 99)
    imerg_avg, imerg_95th, imerg_99th = np.mean(imerg_data), np.percentile(imerg_data, 95), np.percentile(imerg_data, 99)

    # Determine combined bins
    combined_min =  -4.2454214e-17 #min(np.min(giss_data), np.min(imerg_data))
    combined_max = 537.83997 #max(np.max(giss_data), np.max(imerg_data))
    bins = np.linspace(combined_min, combined_max, 50)  # 50 bins

    # Plot histograms with the same bins
    plt.hist(imerg_data, bins=bins, log=True, density=True, alpha=0.5, color='blue', label='IMERG')
    plt.hist(giss_data, bins=bins, log=True, density=True, alpha=0.5, color='red', label='GISS')

    # Add percentile lines with labels for the legend
    plt.axvline(x=giss_99th, color='red', linestyle='dotted', label='GISS 99th percentile')
    plt.axvline(x=imerg_99th, color='blue', linestyle='dotted', label='IMERG 99th percentile')

    # Plot settings
    title = "GISS vs IMERG Precipitation | " + title
    plt.title(title)
    plt.xlabel('Precipitation (mm/day)')
    plt.ylabel('Density (log scale)')
    plt.legend()

    plt.xlim(0, 550)
    plt.ylim(10**-5.5, 1e-1)

    # Add average text on the right side of the plot
    plt.text(0.66, 0.53, f'GISS Avg: {giss_avg:.2f}\nIMERG Avg: {imerg_avg:.2f}\n\nGISS 95th %: {giss_95th:.2f}\nIMERG 95th %: {imerg_95th:.2f}\n\nGISS 99th %: {giss_99th:.2f}\nIMERG 99th %: {imerg_99th:.2f}', 
             horizontalalignment='left', verticalalignment='center', 
             transform=plt.gca().transAxes, color='black', fontsize=7)

    # Show plot for visualization in this context
    plt.show()

    # # Uncomment the following lines if saving the plot is desired
    # plt.savefig(output_path, dpi=300)
    # print(f"Saved plot to {output_path}.")
    plt.close()

def statsTableGISSIMERG(giss_data,imerg_data,title,output_path):
	pass

#----------------------------------END OF FUNCTIONS--------------------------------#


#-------------------------------MAIN CODE-----------------------------------#
if __name__ == "__main__":
	years = list(range(start_year, end_year+1))
	createCompareViz(mode, years,regions, variable_name, GISS_data_folder, IMERG_data_folder, output_folder, chosen_season, months_list) 

#---------------------------------END OF MAIN CODE---------------------------------#
