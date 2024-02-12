import numpy as np
import pickle

def parse_csv(csv_path, interval, file_names, time_slot_duration):
    HERE_PATH = csv_path + 'inflow'
    THERE_PATH = csv_path + 'outflow'
    
    # Initialize structure
    interval_data = []

    for file_name in file_names:
        # Initialize time slot sums for this file
        tor_sums = {}
        exit_sums = {}

        # Process HERE_PATH (inflow)
        with open(f"{HERE_PATH}/{file_name}") as f:
            for line in f:
                time, size = map(float, line.split('\t'))
                if interval[0] <= time <= interval[1]:
                    slot_index = int((time - interval[0]) / time_slot_duration)
                    if slot_index not in tor_sums:
                        tor_sums[slot_index] = 0
                    tor_sums[slot_index] += size

        # Process THERE_PATH (outflow)
        with open(f"{THERE_PATH}/{file_name}") as f:
            for line in f:
                time, size = map(float, line.split('\t'))
                if interval[0] <= time <= interval[1]:
                    slot_index = int((time - interval[0]) / time_slot_duration)
                    if slot_index not in exit_sums:
                        exit_sums[slot_index] = 0
                    exit_sums[slot_index] += size

        # Merge the processed data for this file into a single structure
        max_index = max(max(tor_sums.keys(), default=0), max(exit_sums.keys(), default=0))
        for i in range(max_index + 1):
            interval_data.append([tor_sums.get(i, 0), -exit_sums.get(i, 0)])

    return interval_data

def create_pickle_with_detailed_temporal_patterns(csv_path, file_list, pickle_output_path, interval, time_slot_duration):
    file_names = [line.strip() for line in open(file_list, 'r')]
    
    interval_data = parse_csv(csv_path, interval, file_names, time_slot_duration)
    
    # Print the first 5 elements of interval data
    print("First 5 elements of interval data:", interval_data[:5])
    
    # Print the total number of [inflow, outflow] pairs (intervals)
    print(f"Total [inflow, outflow] pairs (intervals) considered: {len(interval_data)}")
    
    # Save the data to a pickle file
    with open(pickle_output_path + '_interval_based.pickle', 'wb') as handle:
        pickle.dump(interval_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Data saved to pickle with interval-based representation.")

# Example usage with updated time_slot_duration
data_path = '/home/tc9614/CrawlE_Proc/'
file_list_path = '/home/tc9614/TranoptiFlow/output.txt'
prefix_pickle_output = '/home/tc9614/TranoptiFlow/interval_based_representation'
time_slot_duration = 0.05  # Updated duration of each time slot in seconds
interval = (0, float('inf'))  # Adjust the interval as needed

create_pickle_with_detailed_temporal_patterns(data_path, file_list_path, prefix_pickle_output, interval, time_slot_duration)
