import os
import glob
import numpy as np
import pandas as pd

import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt, find_peaks

ppg_data_path = "C:/Users/peiha/Desktop/LP-BP\BP-Analysis/Data/LP_Output"
abp_data_path = "C:/Users/peiha/Desktop/LP-BP\BP-Analysis/Data/Mayooutput"
training_data_path = "C:/Users/peiha/Desktop/LP-BP/data_for_training"

lowcut_freq = 0.5
highcut_freq = 4.0
sample_rate_ppg = 25
sample_rate_abp = 997
sample_length_per_frame = 30

###########################[Function Section]###########################
# Function to insert zeros into specific indices
def insert_zeros(data, indices, number_of_zeros):
    offset = 0
    data_copy = data.copy()
    for index, zeros in zip(indices, number_of_zeros):
        data_copy[index + offset:index + offset] = [0] * zeros
        offset += zeros
    return data_copy


# Function to extract peaks and valleys of the input signal
def peaks_valleys_est(input_signal):
    no_of_peaks, _ = find_peaks(input_signal)
    if len(no_of_peaks) > 0:
        peak_distance = int(len(input_signal) / len(no_of_peaks)) - 1
    else:
        peak_distance = 1
    valleys, _ = find_peaks(np.negative(input_signal), distance=peak_distance)
    peaks, _ = find_peaks(input_signal[valleys[1]:valleys[-2]], distance=peak_distance)
    return peak_distance, peaks + valleys[1], valleys


# Function to find episodes divided by 0s
def find_episodes_divided_by_zeros(df):
    episodes = []
    start = None
    for i in range(len(df)):
        # Transition from 0 to non-zero, indicating start of an episode
        if df.iloc[i] != 0 and (i == 0 or df.iloc[i - 1] == 0):
            start = i
        # Transition from non-zero to 0, indicating end of an episode
        elif df.iloc[i] == 0 and i > 0 and df.iloc[i - 1] != 0:
            end = i - 1
            if start is not None:
                episodes.append((start, end))
                start = None
    # Check for an episode that doesn't end with 0
    if start is not None and df.iloc[-1] != 0:
        episodes.append((start, len(df) - 1))
    return episodes


# Bandpass Filter with order = 4
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


# Project the 'to_compare' points to the nearest 'reference' points
def sort_by_proximity(to_compare, reference):
    reference = np.array(reference)
    to_compare = np.array(to_compare)
    closest_elements = []

    for item in reference:
        closest_element_index = np.argmin(np.abs(to_compare - item))
        closest_elements.append(to_compare[closest_element_index])

    return closest_elements


# Extract cycles that each has two valleys and a peak
def find_peaks_between_valleys(valley, peak):
    """
    This function iterates through the valley array to define ranges and then checks
    for any peaks that fall within each range.

    Parameters:
    - valley: A list of valley points.
    - peak: A list of peak points.

    Returns:
    - A list of dictionaries, each containing the valley range and the peaks found within that range.
    """
    result = []
    for i in range(len(valley) - 1):
        current_peaks = [p for p in peak if valley[i] < p < valley[i + 1]]
        if len(current_peaks) < 2:
            result.append(((valley[i], valley[i + 1]), current_peaks))
    return result


# Pair the PPG cardiac cycles and the ABP cardiac cycles based on the closest peaks
def pair_by_closest_peak(ppg, abp):
    """
    Adjusts PPG valley ranges to match the peak-to-valley distances of their closest ABP peak,
    while keeping ABP ranges unchanged.

    Parameters:
    - ppg: A list of tuples, where each tuple contains a valley range and the peaks within that range.
    - abp: A list of tuples, similar to ppg, for another set of valley ranges and peaks.

    Returns:
    - A list of tuples, where each tuple contains an adjusted PPG pair and the original ABP pair
      based on the closest peak and ABP's peak-to-valley distance.
    """
    paired = []

    for ppg_range, ppg_peaks in ppg:
        closest_distance = float('inf')
        closest_pair = None

        if not ppg_peaks:
            continue

        for abp_range, abp_peaks in abp:
            if abp_peaks:
                distance = abs(ppg_peaks[0] - abp_peaks[0])
                if distance < closest_distance:
                    closest_distance = distance
                    closest_pair = (abp_range, abp_peaks)

        if closest_pair:
            abp_range, abp_peaks = closest_pair
            # Calculate ABP peak-to-valley distances
            abp_distance_start = abp_peaks[0] - abp_range[0]
            abp_distance_end = abp_range[1] - abp_peaks[0]

            # Adjust PPG range to match ABP peak-to-valley distances
            adjusted_ppg_range = (ppg_peaks[0] - abp_distance_start, ppg_peaks[0] + abp_distance_end)

            #             paired.append(((adjusted_ppg_range, ppg_peaks), (abp_range, abp_peaks)))
            paired.append(((ppg_range, ppg_peaks), (abp_range, abp_peaks)))

    return paired


def shift_series_with_offset(series, offset):
    """
    Shifts a pandas Series by the specified offset. For positive offsets, it prepends zeros.
    For negative offsets, it shifts the series' start forward without losing values, and adds zeros at the end.

    Parameters:
    - series: pd.Series, the original pandas Series to be shifted.
    - offset: int, the number of positions to shift the series. Positive, negative, or zero.

    Returns:
    - pd.Series: The shifted series with original values preserved and indices adjusted accordingly.
    """
    if offset > 0:
        # Handle positive offset by prepending zeros
        new_values = [0.0] * offset + series.tolist()
        new_index = range(series.index[0], series.index[-1] + 1 + offset)
    elif offset < 0:
        # Handle negative offset by keeping the series length the same, shifting indices,
        # and appending zeros at the end
        new_values = series.tolist() + [0.0] * abs(offset)
        new_index = range(series.index[0] + offset, series.index[-1] + 1)
    else:
        # If offset is zero, return the series as is
        return series

    return pd.Series(new_values, index=new_index)


if __name__ == "__main__":

    # Extract PPG data and find the gap indices
    for file_path_ppg in glob.glob(f'{ppg_data_path}/*.csv'):

        file_name_ppg = os.path.basename(file_path_ppg)
        ppg_in_25_hz = pd.read_csv(file_path_ppg)

        ###########################[Step 1: In PPG data folder, skip the files that are empty or have reversed timestamps]###########################
        if ppg_in_25_hz.empty:
            print("")
            print(f'PPG data in ###{file_name_ppg}### is empty! Skip this subject.')
            continue

        # Find the gap in the timestamp and return the gap_indices and gap_times
        time_diff = ppg_in_25_hz['timestamps'].diff()

        flag_negative_time_gap = 0
        for i in time_diff:
            if i < 0:
                flag_negative_time_gap = 1
                break

        if flag_negative_time_gap == 1:
            print("")
            print(f'PPG data in ###{file_name_ppg}### has a reversed timestamp! Skip this subject.')
            continue

        ###########################[Step 2: Find timestamp gaps in PPG data and fill the gap with 0s]###########################
        # Any adjacent samples that has a gap larger than 100 second will be tagged
        gap_threshold = 100
        gap_indices = time_diff[time_diff > gap_threshold].index
        gap_times = time_diff[time_diff > gap_threshold]
        gap_samples = round(gap_times * sample_rate_ppg)

        gap_indices_list = gap_indices.tolist()
        gap_samples_list = gap_samples.tolist()
        ppg_list = ppg_in_25_hz['ppg'].tolist()

        # Fill the gap with 0s that the number is equal to the corresponding gap time
        number_of_zeros_list = [int(x) + 250 for x in gap_samples_list]
        number_of_zeros_list.insert(0, 250)
        gap_indices_list.insert(0, 0)

        ppg_filled_with_zeros = insert_zeros(ppg_list, gap_indices_list, number_of_zeros_list)

        ###########################[Step 3: In the ABP data folder, find the file has same name with PPG]###########################
        ###########################[If no such file exists, then skip this round]###########################
        # Extract ABP data from the .csv with the same name (Subject**)
        abp_file_exist_flag = 0
        for file_path_abp in glob.glob(f'{abp_data_path}/*.csv'):
            file_name_abp = os.path.basename(file_path_abp)
            if file_name_abp == file_name_ppg:
                abp_file_exist_flag = 1
                break

        if abp_file_exist_flag != 1:
            print(f'ABP file missing for {file_name_ppg}')
            continue

        abp_in_997_hz = pd.read_csv(file_path_abp)

        ###########################[Step 4: Use interpolation to degrade ABP sampling rate to 25Hz]###########################
        # Convert timestamps to datetime
        abp_in_997_hz['Timestamps'] = pd.to_datetime(abp_in_997_hz['Timestamps'], unit='s')
        # Set 'Timestamps' as the DataFrame index
        abp_in_997_hz = abp_in_997_hz.set_index('Timestamps')
        # Drop the unnamed index column as it's not needed
        abp_in_997_hz.drop(columns=abp_in_997_hz.columns[0], inplace=True)
        # Generate new timestamps for 25 Hz sampling rate
        # Calculate the period for 25 Hz sampling rate in nanoseconds (1e9 nanoseconds = 1 second)
        period_ns = int(1e9 / sample_rate_ppg)

        # Generate new timestamps ranging from the first to the last timestamp of the original data
        new_timestamps = pd.date_range(start=abp_in_997_hz.index[0], end=abp_in_997_hz.index[-1],
                                       freq=f'{period_ns}ns')
        # Interpolate the original data to these new timestamps
        abp_in_25_hz = abp_in_997_hz.reindex(abp_in_997_hz.index.union(new_timestamps)).interpolate(
            method='time').reindex(new_timestamps)

        abp_25Hz_list = abp_in_25_hz['ART'].tolist()

        ###########################[Step 5: Combine the ABP and PPG data into one dataframe]###########################
        # Combine the PPG and ABP into one dataframe with the length of the shortest one
        combined = list(zip(ppg_filled_with_zeros, abp_25Hz_list))
        ppg_with_abp_25_hz = pd.DataFrame(combined, columns=['ppg', 'abp'])

        # Discard 0s in the ppg, only keep the first 250 to align with the abp
        # Step 1: Identify zero values
        ppg_with_abp_25_hz['is_zero'] = ppg_with_abp_25_hz['ppg'] == 0
        # Step 2: Create a group identifier for consecutive zeros
        ppg_with_abp_25_hz['group'] = ppg_with_abp_25_hz['is_zero'].ne(
            ppg_with_abp_25_hz['is_zero'].shift()).cumsum()
        # Step 3: Within each group of consecutive zeros, assign a row number
        ppg_with_abp_25_hz['rank'] = ppg_with_abp_25_hz.groupby('group').cumcount() + 1
        # Step 4: Filter rows. Keep where rank <= 250 and is_zero, or where not is_zero
        filtered_df = ppg_with_abp_25_hz[((ppg_with_abp_25_hz['is_zero'] & (ppg_with_abp_25_hz['rank'] <= 250)) | (
            ~ppg_with_abp_25_hz['is_zero']))].drop(['is_zero', 'group', 'rank'], axis=1)
        filtered_df.reset_index(drop=True)

        ###########################[Step 6: Extract episodes divided by 0s]###########################
        scaler = StandardScaler()  # Initialize MinMaxScaler
        episodes = find_episodes_divided_by_zeros(ppg_with_abp_25_hz['ppg'])
        # print(f'Episode: {len(episodes)}, (From, To): {episodes[0]}')
        print("")
        print(f'### Entering {file_name_abp} processing! Samples: {len(filtered_df)}, Total Episodes: {len(episodes)} ###')

        episode_count = 0
        episode_discard_count = 0
        for episode in episodes:
            episode_count += 1
            print(f'Now processing Episode: {episode_count}, (From, To): {episode}')

            ppg_abp_episode = ppg_with_abp_25_hz.iloc[episode[0]:episode[1] + 1].copy()
            ppg_abp_episode.reset_index(drop=True, inplace=True)
            ppg_abp_episode['ppg_normalized'] = scaler.fit_transform(ppg_abp_episode[['ppg']])
            ppg_abp_episode['ppg_filtered'] = butter_bandpass_filter(ppg_abp_episode['ppg_normalized'],
                                              lowcut_freq, highcut_freq, sample_rate_ppg, order=4)

            #############[Step 7: In each episode, extract every cardiac cycle for both PPG and ABP]#############
            # Find peaks and valleys for both PPG and ABP
            try:
                _, ppg_peaks, ppg_valleys = peaks_valleys_est(ppg_abp_episode['ppg_filtered'])
                _, abp_peaks, abp_valleys = peaks_valleys_est(ppg_abp_episode['abp'])
            except Exception as e:
                print(f"An error occurred: {e}, skip this episode!")
                continue

            # Apply Savitzky-Golay filter for smoothing on the ABP signal to eliminate the Dicrotic Notch
            window_size = 9  # Should be odd; adjust based on your signal's characteristics
            poly_order = 2  # Polynomial order; adjust based on your needs
            abp_smoothed = savgol_filter(ppg_abp_episode['abp'], window_size, poly_order)
            _, smoothed_abp_peaks, smoothed_abp_valleys = peaks_valleys_est(abp_smoothed)

            # Use proximity to sort out the "real" peaks and valleys in the ABP signal
            abp_peaks_without_notch = sort_by_proximity(abp_peaks, smoothed_abp_peaks)
            abp_valleys_without_notch = sort_by_proximity(abp_valleys, smoothed_abp_valleys)

            ###########################[Step 8: Check the error ratio between number of ABP peaks and PPG peaks]###########################
            ###########################[If the error > 20, then skip this episode]###########################
            #ppg_abp_error_ratio = abs(len(ppg_peaks) - len(abp_peaks_without_notch)) / len(abp_peaks_without_notch)
            ppg_abp_abs_error = abs(len(ppg_peaks) - len(abp_peaks_without_notch))
            if ppg_abp_abs_error > 20:
                episode_discard_count += 1
                print(f"""Current episode: {episode_count}, (From, To): {episode}, Discarded for high PPG({len(ppg_peaks)}) and ABP({len(abp_peaks_without_notch)}) error: {ppg_abp_abs_error}""")
                continue
            # else:
            #     print(f'PPG and ABP peaks error ratio: {ppg_abp_error_ratio:.2f}')

            ###########################[Step 9: Pair the PPG cardiac cycles and ABP cardiac cycles]###########################
            ppg_cardiac_cycles = find_peaks_between_valleys(ppg_valleys, ppg_peaks)
            abp_cardiac_cycles = find_peaks_between_valleys(abp_valleys_without_notch,
                                                            abp_peaks_without_notch)
            paired_result = pair_by_closest_peak(ppg_cardiac_cycles, abp_cardiac_cycles)

            cycle_count = 0  # For cardiac cycle counting
            # Define a DataFrame that stores the output
            df_cardiac_cycle = pd.DataFrame(columns=['cycle_index',
                                                     'ppg_begin_peak_end', 'abp_begin_peak_end',
                                                     'correlation',
                                                     'systolic_bp', 'diastolic_bp',
                                                     'ppg_frequency', 'ppg_phase', 'ppg_amplitude'])

            #############[Step 10: In each paired cardiac cycle, align the peak and calculate the correlation]#############
            #############[If the correlation is smaller than 0.39, then discard this pair of PPG and ABP]#############
            # Iterate through all matched cardiac cycles
            for (ppg_range, ppg_peak), (abp_range, abp_peak) in paired_result:
                cycle_count += 1

                # Check if the current cardiac cycle is normal (0.4 - 3 sec)
                bp_cycle_interval = abp_range[1] - abp_range[0]
                if bp_cycle_interval > 75 or bp_cycle_interval < 10:
                    #print(f"Current BP cycle's total samples [{bp_cycle_interval}] is not in the range (10,75), discard this cycle!")
                    continue

                # Extract the systolic bp and diastolic bp
                systolic_bp = ppg_abp_episode['abp'][abp_peak[0]]
                diastolic_bp = min(ppg_abp_episode['abp'][abp_range[0]], ppg_abp_episode['abp'][abp_range[1]])
                systolic_bp = np.round(systolic_bp, 2)
                diastolic_bp = np.round(diastolic_bp, 2)

                # Check if the systolic bp is normal (80 - 180 mmHg)
                if systolic_bp > 180 or systolic_bp < 80:
                    #print(f"[Cycle {cycle_count}]Current SBP ({systolic_bp}) is not in the range (80, 180), discard this cycle!")
                    continue

                # Check if the diastolic bp is normal (>= 20 mmHg)
                if diastolic_bp < 20:
                    #print(f"[Cycle {cycle_count}]Current DBP ({diastolic_bp}) is smaller than 20, discard this cycle!")
                    continue

                # Check if the error between systolic and diastolic bp is normal (< 20 mmHg)
                if systolic_bp - diastolic_bp < 20:
                    #print(f"SBP - DBP < 20, discard this cycle!")
                    continue

                # Check if the current cardiac cycle has a longer begin-to-peak samples than samples-per-frame
                #if ppg_peak[0] - ppg_range[0] > sample_length_per_frame:
                    # print(f"Cycle {cycle_count} has length of {ppg_range[1] - ppg_range[0]}! Skip this round!")
                #    continue

                ppg_current_cycle = ppg_abp_episode['ppg_filtered'][ppg_range[0]:ppg_range[1]]
                abp_current_cycle = ppg_abp_episode['abp'][abp_range[0]:abp_range[1]]

                ppg_offset = abp_peak[0] - ppg_peak[0]
                ppg_shifted_cycle = shift_series_with_offset(ppg_current_cycle, ppg_offset)

                ppg_abp_correlation = ppg_shifted_cycle.corr(abp_current_cycle).round(2)

                # If the current cardiac cycle of ABP and PPG has a correlation less than 0.39, then skip this round
                if ppg_abp_correlation < 0.39:
                    #print(f"Correlation {ppg_abp_correlation} < 0.39, discard this cycle!")
                    continue

                # Extend the PPG cycle with 10% of the previous cycle and 5% of the following cycle
                ppg_current_cycle_length = len(ppg_current_cycle)
                extend_length_previous = round(ppg_current_cycle_length * 0.1)
                extend_length_following = round(ppg_current_cycle_length * 0.05)
                ppg_current_cycle_exteded = ppg_abp_episode['ppg_filtered'][(ppg_range[0] - extend_length_previous):(ppg_range[1] + extend_length_following)]
                #print(f'Original PPG cycle length vs. extended PPG cycle length: {ppg_current_cycle_length} vs. {len(ppg_current_cycle_exteded)}')

                ###########################[Step 11: Get the positive frequency and phase of the ppg signal]###########################
                # Get the frequency and phase of the current ppg cardiac cycle
                if len(ppg_current_cycle_exteded) < sample_length_per_frame:
                    difference = sample_length_per_frame - len(ppg_current_cycle_exteded)
                    ppg_current_frame = pd.concat([ppg_current_cycle_exteded, pd.Series([0] * difference)], ignore_index=True)
                else:
                    ppg_current_frame = ppg_current_cycle_exteded[:sample_length_per_frame]


                fft_ppg_result = np.fft.fft(ppg_current_frame)
                fft_ppg_freq = np.fft.fftfreq(ppg_current_frame.size, d=1 / sample_rate_ppg)
                fft_ppg_magnitude = np.abs(fft_ppg_result)
                fft_ppg_phase = np.angle(fft_ppg_result)

                # Filter out the components with frequencies greater than 0
                positive_freq_indices = np.where(fft_ppg_freq > 0)[0]
                # Select the frequencies greater than 0
                positive_frequencies = fft_ppg_freq[positive_freq_indices]
                # Select the corresponding magnitudes
                positive_magnitudes = fft_ppg_magnitude[positive_freq_indices]
                # Select the corresponding phases
                positive_phases = fft_ppg_phase[positive_freq_indices]

                positive_frequencies = np.round(positive_frequencies, 4)
                positive_magnitudes = np.round(positive_magnitudes, 4)
                positive_phases = np.round(positive_phases, 4)

                positive_frequencies_string = ":".join(map(str, positive_frequencies))
                positive_magnitudes_string = ":".join(map(str, positive_magnitudes))
                positive_phases_string = ":".join(map(str, positive_phases))

                ###########################[Step 12: Extract key information and save it into a "SubjectXX.csv" file]###########################
                data_episode = {
                    'cycle_index': cycle_count,
                    'ppg_begin_peak_end': (ppg_range[0], *ppg_peak, ppg_range[1]),
                    'abp_begin_peak_end': (abp_range[0], *abp_peak, abp_range[1]),
                    'correlation': ppg_abp_correlation,
                    'systolic_bp': systolic_bp,
                    'diastolic_bp': diastolic_bp,
                    'ppg_frequency': positive_frequencies_string,
                    'ppg_phase': positive_phases_string,
                    'ppg_amplitude': positive_magnitudes_string
                }

                df_cardiac_cycle = df_cardiac_cycle._append(data_episode, ignore_index=True)

            # Check if the directory exists, if not, create it
            file_name_text_only = os.path.splitext(file_name_ppg)[0]
            output_folder_path = os.path.join(training_data_path, file_name_text_only)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            # Create and write to the CSV file
            filename = f'episode_{episode_count}.csv'
            full_path = os.path.join(output_folder_path, filename)
            #df_cardiac_cycle.to_csv(full_path)

        print(f"### Ending {file_name_abp} processing! Total episodes: {len(episodes)}, discarded episodes: {episode_discard_count}, discard ratio: {(episode_discard_count / len(episodes)):.2f} ###")
        print("")