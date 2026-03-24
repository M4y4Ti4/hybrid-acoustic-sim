
import os
import sys 
import numpy as np

HYBRID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#adding both submodules to the path
sys.path.append(os.path.join(HYBRID_DIR, "DGsim", "edg-acoustics"))
sys.path.append(os.path.join(HYBRID_DIR, "RayroomProject", "rayroom"))

from rayroom.core.data_anal import overlay_DG, plot_rir, plot_transfer_function


"""
test calibration with loading previously obtained results for the shoebox !

"""


#importing data (saved locally in og repos) => this calibration will eventually be called in pipeline 
#both the wave and geo models have been simulated using the same initial room conditions 

wave_data = np.load(r"C:\Masters\DGsim\edg-acoustics\examples\shoebox\output\corrected_shoebox_lc1_200Hz_2slc__10.npz")

geo_data = np.load(r"C:\Masters\RayroomProject\rayroom\examples\initial_testing\rir_shoebox_cal.npz")


wave_data_path = r"C:\Masters\DGsim\edg-acoustics\examples\shoebox\output\corrected_shoebox_lc1_200Hz_2slc__10.npz"

#loading geometric IR data
rir_total = geo_data["rir_total"]
td = geo_data["t_d"]
tr = geo_data["t_r"]
amp_direct = geo_data["amp_direct"]
amp_first = geo_data["amp_first"]
t_geo = np.linspace(0, 2.0, len(rir_total))


#loading wave IR data
t_wave = wave_data["t_resampled"]
IR = wave_data["IR_cor_resampled"]

overlay_DG(rir_total, wave_data_path, fs=44100)

#access time of arrival, amplitude... of direct sound and first reflection from ism (save this in the output/return)
#scale the GA IR according to equation 
#low pass the GA IR so that contains same frequencies as wave 
#use td and tr to calcaulate w1 and w1 
#calculate calibration parameter using ratio of max value of g and w across w1 and w2

def scale_Gir(gd, td, rir_total, c = 343.0):
    """
    param gd: the amplitude of direct sound from ISM 
    param rir_total: the filtered and summed rir_bands
    param td: the time of arrival of the direct sound
    param c: speed of sound

    """
    rir_scaled = (rir_total * gd)/(c * td) #scaling the geometric RIR
    return rir_scaled 

def low_pass_filter(rir, cutoff, fs, order=4): 
    """
    design a low-pass butterworth filter (to be applied to geometric and wave rir)
    param cutoff: cutoff frequency 
    param fs: sampling frequency
    param order: order of the filter
    param rir: rir to be filtered
    """

    from scipy.signal import butter, sosfilt
    if cutoff <= 0 or fs <= 0:
        raise ValueError("values must be positive")
    if cutoff >= fs / 2: 
        raise ValueError("cutoff freq must be less than Nyquist freq")
    
    nyquist = 0.5 * fs 
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype = 'low', output = 'sos')
    rir_filtered = sosfilt(sos, rir)
    return rir_filtered


def find_max(signal, time, start_time, end_time):
    """
    calculating the maximum of a signal within a time window
    """
    signal = np.array(signal, dtype=float)
    timestamps = np.array(time, dtype=float)

    if signal.shape != timestamps.shape: 
        raise ValueError("Signal and time array must be the same size")
    mask = (timestamps >= start_time) & (timestamps <= end_time)
    window_values = signal[mask]

    if window_values.size == 0:
        return None
    return np.max(window_values)

def calculate_calibration(gd, td, tr, geo_rir, wave_rir, t_wave, t_geo):
    """
    calculating the calibration coefficient
    """
    w1 = td - 0.5*(tr - td)
    w2 = td + 0.5*(tr - td)
    
    max_wave = find_max(wave_rir, t_wave, w1, w2)
    max_geo = find_max(geo_rir, t_geo, w1, w2)

    cal_coef = max_geo/max_wave

    return cal_coef

rir_g_scaled = scale_Gir(gd = amp_direct, td=td, rir_total = rir_total)
rir_g_filtered = low_pass_filter(rir_g_scaled, 200, fs = 44100)
rir_w_filtered = low_pass_filter(IR, 200, fs = 44100)

#plot_rir(rir_total, fs = 44100)
#plot_rir(rir_g_scaled, fs = 44100)
#plot_rir(rir_g_filtered, fs = 44100)
#plot_rir(rir_w_filtered, fs = 44100)

plot_transfer_function(rir_g_filtered, fs = 44100)

eta = calculate_calibration(gd=amp_direct, td=td, tr = tr, geo_rir = rir_g_filtered, wave_rir = rir_w_filtered, t_wave = t_wave, t_geo = t_geo)


