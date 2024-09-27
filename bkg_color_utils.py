from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_histograms_with_ranges(path, hist_L, hist_A, hist_B, range_L, range_A, range_B):
    plt.figure(figsize=(15, 5))

    # Plot for L channel
    plt.subplot(1, 3, 1)
    plt.plot(hist_L, color='black')
    plt.axvline(x=range_L[0], color='red', linestyle='--', label=f'Range Start: {range_L[0]}')
    plt.axvline(x=range_L[1], color='green', linestyle='--', label=f'Range End: {range_L[1]}')
    plt.title('L Channel Histogram')
    plt.legend()

    # Plot for A channel
    plt.subplot(1, 3, 2)
    plt.plot(hist_A, color='black')
    plt.axvline(x=range_A[0], color='red', linestyle='--', label=f'Range Start: {range_A[0]}')
    plt.axvline(x=range_A[1], color='green', linestyle='--', label=f'Range End: {range_A[1]}')
    plt.title('A Channel Histogram')
    plt.legend()

    # Plot for B channel
    plt.subplot(1, 3, 3)
    plt.plot(hist_B, color='black')
    plt.axvline(x=range_B[0], color='red', linestyle='--', label=f'Range Start: {range_B[0]}')
    plt.axvline(x=range_B[1], color='green', linestyle='--', label=f'Range End: {range_B[1]}')
    plt.title('B Channel Histogram')
    plt.legend()

    plt.savefig(path)

def get_space_range_using_histograms(frame, view_dir, write_video=False):
        
    lab_image = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    lab_mask = cv2.inRange(lab_image, (1, 0, 0), (255, 255, 255))
    
    # Calculate the histogram for each channel (L, A, B)
    hist_L = cv2.calcHist([lab_image], [0], lab_mask, [256], [0, 256])
    hist_A = cv2.calcHist([lab_image], [1], lab_mask, [256], [0, 256])
    hist_B = cv2.calcHist([lab_image], [2], lab_mask, [256], [0, 256])

    # Function to find the range covering 90% of the pixels
    def find_percent_range(hist, percentage):
        cdf = hist.cumsum()  
        total_pixels = cdf[-1]  
        peak_value = np.argmax(hist)  # Index of the peak value

        # Find the range around the peak that covers 90% of the pixels
        lower_bound = peak_value
        upper_bound = peak_value
        covered_pixels = hist[peak_value]

        while covered_pixels / total_pixels < percentage:
            if lower_bound > 0:
                lower_bound -= 1
            if upper_bound < len(hist) - 1:
                upper_bound += 1
            covered_pixels = cdf[upper_bound] - cdf[lower_bound]
        
        return lower_bound, upper_bound

    # Find the coverage range for each channel
    percentage = 0.85
    range_L_start, range_L_end = find_percent_range(hist_L,percentage)
    range_A_start, range_A_end = find_percent_range(hist_A,percentage)
    range_B_start, range_B_end = find_percent_range(hist_B,percentage)
    if write_video:
        plot_histograms_with_ranges( os.path.join(view_dir, 'bkg_histograms.png'),
                                    hist_L, hist_A, hist_B, 
                                    (range_L_start, range_L_end),
                                    (range_A_start, range_A_end), 
                                    (range_B_start, range_B_end))
    
    return range_L_start, range_L_end, range_A_start, range_A_end, range_B_start, range_B_end