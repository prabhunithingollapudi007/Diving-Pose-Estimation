import numpy as np
from scipy.ndimage import gaussian_filter1d

def moving_average_filter(data, window_size):
    """Applies a moving average filter to the input data.

    Args:
        data: A NumPy array of data points.
        window_size: The number of data points to include in the average.

    Returns:
        A NumPy array of filtered data.
    """
    if window_size <= 0 or window_size > len(data):
        raise ValueError("Invalid window size.")

    filtered_data = np.convolve(data, np.ones(window_size) / window_size, mode='same')
    return filtered_data

def gaussian_filter(data, sigma):
    """Applies a Gaussian filter to the input data.

    Args:
        data: A NumPy array of data points.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        A NumPy array of filtered data.
    """
    filtered_data = gaussian_filter1d(data, sigma)
    return filtered_data