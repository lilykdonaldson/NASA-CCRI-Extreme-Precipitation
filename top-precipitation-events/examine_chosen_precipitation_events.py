#This code finds and segments precipitation events in a CSV dataset 
#and filters the found precipitation events by the given dates (defined in the dates array)
#You may want to use find_top_precipitation_events.py to find highest ranked precipitation events in your dataset first.
import pandas as pd

def analyze_precipitation_data(file_paths, dates):
    all_events = pd.DataFrame()
    
    for file_path in file_paths:
        # Read data
        data = pd.read_csv(file_path, skiprows=10, names=['Timestamp', 'Precipitation'])
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])

        # Define events
        data['IsBreak'] = data['Precipitation'] <= 0.89
        data['EventID'] = data['IsBreak'].cumsum()

        # Total precipitation and duration for each event
        event_totals = data.groupby('EventID').agg(
            Start=pd.NamedAgg(column='Timestamp', aggfunc='first'),
            End=pd.NamedAgg(column='Timestamp', aggfunc='last'),
            TotalPrecipitation=pd.NamedAgg(column='Precipitation', aggfunc='sum')
        )

        # Average precipitation rate (mm/hour)
        event_totals['DurationHours'] = (event_totals['End'] - event_totals['Start']).dt.total_seconds() / 3600
        event_totals['AverageRate'] = event_totals['TotalPrecipitation'] / event_totals['DurationHours']
        
        # Exclude events with zero duration
        event_totals = event_totals[event_totals['DurationHours'] > 0] 

        # Filter events with precipitation
        ranked_events = event_totals[event_totals['TotalPrecipitation'] > 0]

        # Ranking
        ranked_events['TotalPrecipRank'] = ranked_events['TotalPrecipitation'].rank(ascending=False)
        ranked_events['AverageRateRank'] = ranked_events['AverageRate'].rank(ascending=False)

        # Calculate weighted score
        ranked_events['WeightedScore'] = (
            0.45 * ranked_events['TotalPrecipRank'] +
            0.45 * ranked_events['AverageRateRank'] +
            0.10 * ranked_events['TotalPrecipitation'].rank(ascending=False) # Using total precipitation for the 10% weight
        )

        # Add events to the all_events DataFrame
        all_events = pd.concat([all_events, ranked_events], ignore_index=True)
    
    # Merging the event data with the original measurements
    merged_data = data.merge(all_events, on='EventID', how='left')

    # Filter events based on whether they contain a measurement on the specified dates
    filtered_events = merged_data[merged_data['Timestamp'].dt.date.isin(dates)]

    # Group by EventID to get unique events
    unique_filtered_events = filtered_events.groupby('EventID').first()

    # Save the filtered events to CSV
    unique_filtered_events.to_csv('/mnt/data/Filtered_Events_Containing_Dates.csv', index=False)
    return unique_filtered_events

# File paths for each year
file_paths = [f'/Users/lilydonaldson/Downloads/examples/current_scripts/NYC_IMERG_Data/{year}.csv' for year in range(2001, 2023)]


# Dates to filter
dates = [
    '2001-03-13', '2001-03-21', '2001-03-30', '2002-06-06', '2002-10-26', 
    '2003-02-22', '2003-05-26', '2003-06-13', '2003-09-23', '2003-10-29',
    '2004-09-18', '2005-01-14', '2005-03-28', '2006-06-07', '2006-09-01',
    '2006-10-28', '2007-03-02', '2008-02-13', '2008-07-27', '2009-07-23',
    '2011-03-06', '2011-09-06', '2012-04-22', '2014-08-13', '2015-01-18',
    '2015-09-30', '2016-01-10', '2016-02-16', '2017-01-23', '2017-10-29',
    '2018-03-02', '2019-01-24', '2019-10-16', '2020-07-10'
]

# Analyze and get results
filtered_events = analyze_precipitation_data(file_paths, dates)

# Display filtered events
print(filtered_events)