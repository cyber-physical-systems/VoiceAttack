import os
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch
from numpy.fft import fft
from decimal import Decimal
import pandas as pd

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
                print(f"Removed .DS_Store from {root}")

def parse_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        sizes = df['RNN'].values if 'RNN' in df else np.array([])
        times = df['time'].values if 'time' in df else np.array([])
        # print(f"Parsed {file_path}: sizes={sizes}, times={times}")  # Debug print
        return sizes, times
    except Exception as e:
        print(f"An error occurred while parsing the CSV file: {e}")
        return np.array([]), np.array([])


# def calculate_features(sizes, times):
#     if sizes.size == 0 or times.size == 0:
#         print("Empty sizes or times array, returning None")  # Debug print
#         return None
    
#     try:
#         start_time = Decimal(str(times[0]))
#         end_time = Decimal(str(times[-1]))
#         duration = end_time - start_time
#         # print(f"Duration: {duration}")  # Debug print


#         mean = Decimal(str(np.mean(sizes)))
#         # print(f"Mean: {mean}")  # Debug print
#         std_dev = Decimal(str(np.std(sizes, ddof=1)))
#         variance = Decimal(str(np.var(sizes, ddof=1)))
#         # max_value = Decimal(str(np.max(sizes)))
#         # min_value = Decimal(str(np.min(sizes)))
#         # range_value = max_value - min_value
#         median = Decimal(str(np.median(sizes)))
#         mad = Decimal(str(np.mean(np.abs(sizes - np.mean(sizes)))))
#         skewness = Decimal(str(skew(sizes)))
#         kurt = Decimal(str(kurtosis(sizes)))

#         fft_vals = np.abs(fft(sizes))
#         fft_mean = Decimal(str(np.mean(fft_vals)))
#         fft_std_dev = Decimal(str(np.std(fft_vals)))

#         hist, _ = np.histogram(sizes, bins='auto')
#         data_entropy = Decimal(str(entropy(hist)))

#         peaks, properties = find_peaks(sizes)
#         num_peaks = len(peaks)

#         rms = Decimal(str(np.sqrt(np.mean(np.square(sizes)))))
#         sma = Decimal(str(np.mean(np.abs(sizes))))
        
#         if len(sizes) > 1:
#             autocorr_lag_1 = Decimal(str(np.corrcoef(sizes[:-1], sizes[1:])[0, 1]))
#         else:
#             autocorr_lag_1 = Decimal(0)
        
#         crest_factor = Decimal(str(np.max(np.abs(sizes)) / rms))
#         # zero_crossing_indices = np.where(np.diff(np.signbit(sizes)))[0]
#         # zero_crossing_rate = Decimal(str(len(zero_crossing_indices) / float(sizes.size)))
        
#         spectral_centroid = Decimal(str(np.dot(np.arange(len(fft_vals)), fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0))

#         power_spectrum, freqs = welch(sizes, nperseg=1024)
#         spectral_entropy = Decimal(str(entropy(power_spectrum)))
#         energy = np.sum(np.square(sizes))
#         prob_density = np.square(sizes) / energy
#         entropy_of_energy = Decimal(str(entropy(prob_density)))
        
#         # spectral_centroid_value = float(spectral_centroid)
#         spectral_rolloff_threshold = 0.85 * np.sum(power_spectrum)
#         spectral_rolloff = Decimal(str(freqs[np.where(np.cumsum(power_spectrum) >= spectral_rolloff_threshold)[0][0]]))
        
#         spectral_flux = Decimal(str(np.sqrt(np.mean(np.diff(power_spectrum) ** 2))))
#         harmonic_signal = np.abs(np.fft.ifft(np.abs(np.fft.fft(sizes))))
#         thd = Decimal(str(np.sqrt(np.mean((sizes - harmonic_signal) ** 2)) / np.sqrt(np.mean(harmonic_signal ** 2))))

#         snr = Decimal(str(10 * np.log10(np.mean(np.square(sizes)) / np.mean(np.square(sizes - np.mean(sizes))))))
        

#         # Additional features
#         waveform_length = Decimal(str(np.sum(np.abs(np.diff(sizes)))))
#         entropy_packet_distribution = Decimal(str(entropy(sizes + np.finfo(float).eps)))  # Adding eps for stability
#         autocorr_function = np.correlate(sizes - np.mean(sizes), sizes - np.mean(sizes), mode='full')
#         max_autocorr_peak = Decimal(str(np.max(autocorr_function[len(sizes)-1:])))
#         coefficient_of_variation = Decimal(str(np.std(sizes) / np.mean(sizes)))
#         first_diff = np.diff(sizes)
#         first_diff_mean = Decimal(str(np.mean(first_diff)))
#         first_diff_variance = Decimal(str(np.var(first_diff, ddof=1)))
#         cumulative_sum = Decimal(str(np.sum(sizes)))
#         signal_energy = Decimal(str(np.sum(np.square(sizes))))


#         # Advanced Statistical and Signal Processing Features
#         spectral_flatness = Decimal(str(np.exp(np.mean(np.log(power_spectrum + np.finfo(float).eps))) / np.mean(power_spectrum)))
#         spectral_kurtosis = Decimal(str(kurtosis(power_spectrum)))
#         spectral_skewness = Decimal(str(skew(power_spectrum)))
#         smoothness = Decimal(str(np.mean(np.diff(sizes, n=2)**2)))
#     except Exception:
#         return None


#     return {
#         'mean': mean,
#         'std_dev': std_dev,
#         'variance': variance,
#         # 'max_value': max_value,
#         # 'min_value': min_value,
#         # 'range': range_value,
#         'median': median,
#         'mean_absolute_deviation': mad,
#         'skewness': skewness,
#         'kurtosis': kurt,
#         'duration': duration, 
#         'fft_mean': fft_mean,
#         'fft_std_dev': fft_std_dev,
#         'entropy': data_entropy,
#         'num_peaks': num_peaks,
#         'rms': rms,
#         'sma': sma,
#         'autocorrelation_lag_1': autocorr_lag_1,
#         'crest_factor': crest_factor,
#         # 'zero_crossing_rate': zero_crossing_rate,
#         'spectral_centroid': spectral_centroid,
#         'spectral_entropy': spectral_entropy,
#         'entropy_of_energy': entropy_of_energy,
#         'spectral_rolloff': spectral_rolloff,
#         'spectral_flux': spectral_flux,
#         'thd': thd,
#         'snr': snr,

#         # additional
#         'waveform_length': waveform_length,
#         'entropy_packet_distribution': entropy_packet_distribution,
#         'max_autocorrelation_peak': max_autocorr_peak,
#         'coefficient_of_variation': coefficient_of_variation,
#         'first_diff_mean': first_diff_mean,
#         'first_diff_variance': first_diff_variance,
#         'cumulative_sum': cumulative_sum,
#         'signal_energy': signal_energy,

#         # advanced
#         'spectral_flatness': spectral_flatness,
#         'spectral_kurtosis': spectral_kurtosis,
#         'spectral_skewness': spectral_skewness,
#         'smoothness': smoothness

#     }


def calculate_features(sizes, times):
    if sizes.size == 0 or times.size == 0:
        print("Empty sizes or times array, returning None")
        return None

    try:
        start_time = Decimal(str(times[0]))
        end_time = Decimal(str(times[-1]))
        duration = end_time - start_time

        mean = Decimal(str(np.mean(sizes)))

        std_dev = Decimal(str(np.std(sizes, ddof=1)))

        variance = Decimal(str(np.var(sizes, ddof=1)))

        median = Decimal(str(np.median(sizes)))

        mad = Decimal(str(np.mean(np.abs(sizes - np.mean(sizes)))))

        skewness = Decimal(str(skew(sizes)))

        kurt = Decimal(str(kurtosis(sizes)))

        fft_vals = np.abs(fft(sizes))
        fft_mean = Decimal(str(np.mean(fft_vals)))

        fft_std_dev = Decimal(str(np.std(fft_vals)))

        hist, _ = np.histogram(sizes, bins='auto')
        data_entropy = Decimal(str(entropy(hist)))

        peaks, properties = find_peaks(sizes)
        num_peaks = len(peaks)

        rms = Decimal(str(np.sqrt(np.mean(np.square(sizes)))))
        
        sma = Decimal(str(np.mean(np.abs(sizes))))

        if len(sizes) > 1:
            autocorr_lag_1 = Decimal(str(np.corrcoef(sizes[:-1], sizes[1:])[0, 1]))
        else:
            autocorr_lag_1 = Decimal(0)

        try:
            crest_factor = Decimal(str(np.max(np.abs(sizes)) / float(rms)))
        except Exception as e:
            print(f"Error calculating crest_factor: {e}")

        try:
            spectral_centroid = Decimal(str(np.dot(np.arange(len(fft_vals)), fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0))
        except Exception as e:
            print(f"Error calculating spectral_centroid: {e}")

        try:
            power_spectrum, freqs = welch(sizes, nperseg=1024)
            spectral_entropy = Decimal(str(entropy(power_spectrum)))
        except Exception as e:
            print(f"Error calculating spectral_entropy: {e}")

        try:
            energy = np.sum(np.square(sizes))
            prob_density = np.square(sizes) / energy
            entropy_of_energy = Decimal(str(entropy(prob_density)))
        except Exception as e:
            print(f"Error calculating entropy_of_energy: {e}")

        try:
            spectral_rolloff_threshold = 0.85 * np.sum(power_spectrum)
            spectral_rolloff = Decimal(str(freqs[np.where(np.cumsum(power_spectrum) >= spectral_rolloff_threshold)[0][0]]))
        except Exception as e:
            print(f"Error calculating spectral_rolloff: {e}")

        try:
            spectral_flux = Decimal(str(np.sqrt(np.mean(np.diff(power_spectrum) ** 2))))
        except Exception as e:
            print(f"Error calculating spectral_flux: {e}")

        try:
            harmonic_signal = np.abs(np.fft.ifft(np.abs(np.fft.fft(sizes))))
            thd = Decimal(str(np.sqrt(np.mean((sizes - harmonic_signal) ** 2)) / float(rms)))
        except Exception as e:
            print(f"Error calculating thd: {e}")

        try:
            snr = Decimal(str(10 * np.log10(np.mean(np.square(sizes)) / np.mean(np.square(sizes - float(np.mean(sizes)))))))
        except Exception as e:
            print(f"Error calculating snr: {e}")

        try:
            waveform_length = Decimal(str(np.sum(np.abs(np.diff(sizes)))))
        except Exception as e:
            print(f"Error calculating waveform_length: {e}")

        try:
            entropy_packet_distribution = Decimal(str(entropy(sizes + np.finfo(float).eps)))
        except Exception as e:
            print(f"Error calculating entropy_packet_distribution: {e}")

        try:
            autocorr_function = np.correlate(sizes - np.mean(sizes), sizes - np.mean(sizes), mode='full')
            max_autocorr_peak = Decimal(str(np.max(autocorr_function[len(sizes)-1:])))
        except Exception as e:
            print(f"Error calculating max_autocorr_peak: {e}")

        try:
            coefficient_of_variation = Decimal(str(np.std(sizes) / np.mean(sizes)))
        except Exception as e:
            print(f"Error calculating coefficient_of_variation: {e}")

        try:
            first_diff = np.diff(sizes)
            first_diff_mean = Decimal(str(np.mean(first_diff)))
            first_diff_variance = Decimal(str(np.var(first_diff, ddof=1)))
        except Exception as e:
            print(f"Error calculating first_diff_mean or first_diff_variance: {e}")

        try:
            cumulative_sum = Decimal(str(np.sum(sizes)))
        except Exception as e:
            print(f"Error calculating cumulative_sum: {e}")

        try:
            signal_energy = Decimal(str(np.sum(np.square(sizes))))
        except Exception as e:
            print(f"Error calculating signal_energy: {e}")

        try:
            spectral_flatness = Decimal(str(np.exp(np.mean(np.log(power_spectrum + np.finfo(float).eps))) / np.mean(power_spectrum)))
        except Exception as e:
            print(f"Error calculating spectral_flatness: {e}")

        try:
            spectral_kurtosis = Decimal(str(kurtosis(power_spectrum)))
        except Exception as e:
            print(f"Error calculating spectral_kurtosis: {e}")

        try:
            spectral_skewness = Decimal(str(skew(power_spectrum)))
        except Exception as e:
            print(f"Error calculating spectral_skewness: {e}")

        try:
            smoothness = Decimal(str(np.mean(np.diff(sizes, n=2)**2)))
        except Exception as e:
            print(f"Error calculating smoothness: {e}")

    except Exception as e:
        print(f"An error occurred during feature calculation: {e}")
        return None

    return {
        'mean': mean,
        'std_dev': std_dev,
        'variance': variance,
        'median': median,
        'mean_absolute_deviation': mad,
        'skewness': skewness,
        'kurtosis': kurt,
        'duration': duration, 
        'fft_mean': fft_mean,
        'fft_std_dev': fft_std_dev,
        'entropy': data_entropy,
        'num_peaks': num_peaks,
        'rms': rms,
        'sma': sma,
        'autocorrelation_lag_1': autocorr_lag_1,
        'crest_factor': crest_factor,
        'spectral_centroid': spectral_centroid,
        'spectral_entropy': spectral_entropy,
        'entropy_of_energy': entropy_of_energy,
        'spectral_rolloff': spectral_rolloff,
        'spectral_flux': spectral_flux,
        'thd': thd,
        'snr': snr,
        'waveform_length': waveform_length,
        'entropy_packet_distribution': entropy_packet_distribution,
        'max_autocorrelation_peak': max_autocorr_peak,
        'coefficient_of_variation': coefficient_of_variation,
        'first_diff_mean': first_diff_mean,
        'first_diff_variance': first_diff_variance,
        'cumulative_sum': cumulative_sum,
        'signal_energy': signal_energy,
        'spectral_flatness': spectral_flatness,
        'spectral_kurtosis': spectral_kurtosis,
        'spectral_skewness': spectral_skewness,
        'smoothness': smoothness
    }

def process_file(file_path):
    sizes, times = parse_csv(file_path)
    features = calculate_features(sizes, times)
    if features is None:
        features = {key: 'NA' for key in
                     ['mean', 'std_dev', 'variance', 'median', 'mean_absolute_deviation', 'skewness', 'kurtosis', #7
                      'duration', 'fft_mean', 'fft_std_dev', 'entropy', 'num_peaks', 'rms', 'sma', 'autocorrelation_lag_1', #8
                        'crest_factor', 'spectral_centroid', 'spectral_entropy', 'entropy_of_energy', 'spectral_rolloff',  #5
                        'spectral_flux', 'thd', 'snr','waveform_length', 'entropy_packet_distribution', 'max_autocorrelation_peak', #6
                        'coefficient_of_variation', 'first_diff_mean', 'first_diff_variance', 'cumulative_sum', 'signal_energy', #5
                        'spectral_flatness', 'spectral_kurtosis', 'spectral_skewness', 'smoothness']} # 4
    return features

# print(process_file("../../data_processed/vc_200/alexa/resampled_02s/1/1.csv"))

def process_folder(folder_path, label, executor):
    features_list = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv') and is_int(f.split('.')[0])], key=lambda x: int(x.split('.')[0]))
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        future = executor.submit(process_file, file_path)
        features_list.append((future, label))
    return features_list


# def write_features_to_csv(features_list, csv_file_path):
#     with open(csv_file_path, 'w', newline='') as file:
#         writer = None
#         for future, label in features_list:
#             features = future.result()
#             features['label'] = label
#             if writer is None:
#                 # Initialize writer with the keys from the first feature set
#                 writer = csv.DictWriter(file, fieldnames=features.keys())
#                 writer.writeheader()
#             writer.writerow(features)

def write_features_to_csv(features_list, csv_file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
    with open(csv_file_path, 'w', newline='') as file:
        writer = None
        for future, label in features_list:
            features = future.result()
            features['label'] = label
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=features.keys())
                writer.writeheader()
            writer.writerow(features)

# def write_features_to_csv(features_list, csv_file_path):
#     os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    
#     with open(csv_file_path, 'w', newline='') as file:
#         writer = None
#         for future, label in features_list:
#             features = future.result()
#             features['label'] = label
#             # print(f"Features before writing: {features}")  # Debug print
#             if writer is None:
#                 writer = csv.DictWriter(file, fieldnames=features.keys())
#                 writer.writeheader()
#             writer.writerow(features)

def main():
    parser = argparse.ArgumentParser(description="Extract features from .csv files and write to a CSV with labels.")
    parser.add_argument("--csv_folder", required=True, help="Folder containing .csv files in labeled subfolders.")
    parser.add_argument("--csv_file_path", required=True, help="Path to output CSV file.")
    args = parser.parse_args()

    remove_ds_store(args.csv_folder)
    with ProcessPoolExecutor() as executor:
        features_futures = []
        subdirs = [d for d in os.listdir(args.csv_folder) if os.path.isdir(os.path.join(args.csv_folder, d)) and is_int(d)]
        subdirs_sorted = sorted(subdirs, key=lambda x: int(x))
        for subdir in subdirs_sorted:
            subdir_path = os.path.join(args.csv_folder, subdir)
            features = process_folder(subdir_path, subdir, executor)
            features_futures.extend(features)

        write_features_to_csv(features_futures, args.csv_file_path)

if __name__ == "__main__":
    main()

