from atmospkg.calculation import sea_level_pressure, virtual_temperature
from mypkgs.processor.gridmethod import find_nearestgrid_2D, find_minimum_ind_2D

def find_minP_in_window(slp, first_guess, window_length):
    """to find TC center by min sea-level pressure\n
    take a square which first_guess is center and 2*windows_length+1 is side length of the square, then find min slp in that square

    Args:
        slp (ndarray(float)): sea-level pressure
        first_guess ((int, int)): (x, y) index of center of window
        window_length (int): half length of window (unit: index)

    Returns:
        (int, int): (x, y) index of TC center (by min slp) 
    """
    start_y, start_x = first_guess[0], first_guess[1]
    window_zero_point = (start_y-window_length, start_x-window_length)
    slp_in_window = slp[start_y-window_length:start_y+window_length+1, start_x-window_length:start_x+window_length+1]
    ind_in_window = find_minimum_ind_2D(slp_in_window)  # return (y, x)
    TC_center_ind = (ind_in_window[0] + window_zero_point[0], ind_in_window[1] + window_zero_point[1])
    return TC_center_ind # (y, x)