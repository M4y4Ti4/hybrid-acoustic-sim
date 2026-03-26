
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt

HYBRID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#adding both submodules to the path
sys.path.append(os.path.join(HYBRID_DIR, "DGsim", "edg-acoustics"))
sys.path.append(os.path.join(HYBRID_DIR, "RayroomProject", "rayroom"))

from rayroom.core.data_anal import overlay_DG, plot_rir, plot_transfer_function

from rayroom.core.utils import bandpass_filter

"""
test calibration with loading previously obtained results for the shoebox !

"""


#importing data (saved locally in og repos) => this calibration will eventually be called in pipeline 
#both the wave and geo models have been simulated using the same initial room conditions 

wave_data = np.load(r"C:\Masters\DGsim\edg-acoustics\examples\shoebox\output\corrected_shoebox_lc1_200Hz_2slc__10.npz")
#wave_data_path = r"C:\Masters\Hybrid\hybridsim\results\corrected_shoebox_lc1_alpha01_freq200_2s"
#wave_data = np.load(wave_data_path)
print(wave_data)

geo_data = np.load(r"C:\Masters\Hybrid\RayroomProject\examples\initial_testing\rir_shoebox_cal_newscale.npz")
#geo_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\rir_shoebox_constabs.npz")


#wave_data_path = r"C:\Masters\DGsim\edg-acoustics\examples\shoebox\output\corrected_shoebox_lc1_200Hz_2slc__10.npz"

#loading geometric IR data
rir_total = geo_data["rir_total"]
td = geo_data["t_d"]
tr = geo_data["t_r"]
amp_direct = geo_data["amp_direct"]
amp_first = geo_data["amp_first"]
t_geo = np.linspace(0, 2.0, len(rir_total))
#rir_g_raw = geo_data["rir_raw"]
amp_direct_scalar = np.mean(np.abs(np.array(amp_direct))) #taking the mean of the band amplitudes for broadband scaling 


plot_rir(rir_total, fs = 44100)

print(td)

#loading wave IR data
t_wave = wave_data["t_resampled"]
IR = wave_data["IR_cor_resampled"]

overlay_DG(rir_total, wave_rir = IR, fs = 44100)
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


def low_pass_filter(rir, cutoff, fs, order=8): 
    """
    design a low-pass butterworth filter (to be applied to geometric and wave rir)
    param cutoff: cutoff frequency 
    param fs: sampling frequency
    param order: order of the filter
    param rir: rir to be filtered
    """

    from scipy.signal import butter, sosfiltfilt
    if cutoff <= 0 or fs <= 0:
        raise ValueError("values must be positive")
    if cutoff >= fs / 2: 
        raise ValueError("cutoff freq must be less than Nyquist freq")
    
    nyquist = 0.5 * fs 
    normal_cutoff = cutoff / nyquist
    sos = butter(order, normal_cutoff, btype = 'low', output = 'sos')
    rir_filtered = sosfiltfilt(sos, rir)
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


def calculate_cal_coef(td, tr, geo_rir, wave_rir, t_wave, t_geo):
    """
    calculating the calibration coefficient
    """

    w1 = td - 0.5*(tr - td)
    w2 = td + 0.5*(tr - td)
    print(w1, w2)
    max_wave = find_max(wave_rir, t_wave, w1, w2)
    print(max_wave)
    max_geo = find_max(geo_rir, t_geo, w1, w2)
    print(max_geo)

    cal_coef = max_geo/max_wave
    print(f"td = {float(td)*1000:.2f}ms")
    print(f"tr = {float(tr)*1000:.2f}ms")
    print(f"w1 = {float(w1)*1000:.2f}ms")
    print(f"w2 = {float(w2)*1000:.2f}ms")
    print(f"max_wave = {max_wave:.6e}")
    print(f"max_geo  = {max_geo:.6e}")
    print(f"eta = max_wave/max_geo = {max_wave/max_geo:.6f}")

    return cal_coef


def calculate_cal_coef_freqdomain(geo_rir, wave_rir, fs=44100, f_low=50, f_high=180):
    
    # Use the shorter length for both
    n = min(len(geo_rir), len(wave_rir))
    
    F_geo  = np.fft.rfft(geo_rir[:n])
    F_wave = np.fft.rfft(wave_rir[:n])
    freqs  = np.fft.rfftfreq(n, 1/fs)

    mask = (freqs >= f_low) & (freqs <= f_high)
    
    rms_geo  = np.sqrt(np.mean(np.abs(F_geo[mask])**2))
    rms_wave = np.sqrt(np.mean(np.abs(F_wave[mask])**2))

    eta = rms_geo/rms_wave
    print(f"Frequency domain calibration ({f_low}-{f_high}Hz):")
    print(f"  RMS geo:  {rms_geo:.6e}")
    print(f"  RMS wave: {rms_wave:.6e}")
    print(f"  eta:      {eta:.6f}")

    return eta

rir_g_scaled = scale_Gir(gd = amp_direct_scalar, td = td, rir_total=rir_total)

geo_low_pass = low_pass_filter(rir_g_scaled, 200, fs = 44100)
wave_low_pass = low_pass_filter(IR, 200, fs = 44100)

#eta = calculate_cal_coef(td, tr, geo_low_pass, wave_low_pass, t_geo = t_geo, t_wave = t_wave)
eta = calculate_cal_coef_freqdomain(geo_low_pass, wave_low_pass)

wave_calibrated = IR * eta 


overlay_DG(rir_g_scaled, wave_rir = wave_calibrated)

overlay_DG(rir_g_scaled, wave_rir = rir_total)

def apply_crossover(geo_rir, wave_rir, crossover_hz, fs = 44100, butter_order=4):
    """
    apply a linkwitz-riley crossover to split a signal into low and high bands
    input: 
    geo_rir: scaled geometric RIR
    wave_rir: calibrated wave RIR
    """
    from scipy.signal import butter, sosfilt

    sos_low = butter(butter_order, crossover_hz, btype='low', fs = fs, output = 'sos')
    sos_high = butter(butter_order, crossover_hz, btype='high', fs=fs, output='sos')

    low_band = sosfilt(sos_low, sosfilt(sos_low, wave_rir))
    high_band = sosfilt(sos_high, sosfilt(sos_high, geo_rir))

    return low_band, high_band

low_band, high_band = apply_crossover(rir_g_scaled, wave_calibrated, crossover_hz=200)

overlay_DG(high_band, wave_rir = low_band, fs = 44100)


