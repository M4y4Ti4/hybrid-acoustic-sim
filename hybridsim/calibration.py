
import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
from rayroom.analytics.acoustics import schroeder_integration, calculate_rt60, calculate_edt, _calculate_decay_time, calculate_drr


HYBRID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#adding both submodules to the path
sys.path.append(os.path.join(HYBRID_DIR, "DGsim", "edg-acoustics"))
sys.path.append(os.path.join(HYBRID_DIR, "RayroomProject", "rayroom"))

from rayroom.core.data_anal import overlay_DG, plot_rir, plot_transfer_function

from rayroom.core.utils import bandpass_filter

"""
functions to calibrate and combine the geometric and wave results 
"""


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


def high_pass_filter(rir, fs, min_freq, order=4):
    """
    design high-pass butterworth filter (to be applied to geoemtric RIR)
    """
    from scipy.signal import butter, sosfiltfilt

    nyquist = 0.5 * fs
    normal_cutoff = min_freq / nyquist
    sos = butter(order, normal_cutoff, btype = 'high', output = 'sos')
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


def integrate_energy(ir_geo, ir_wav, t_wave, t_geo, t_start, t_end):
    """
    integrate the energy over the early reflections of the impulse response 
    the time of arrival of the first non-specular ray is detected (to indicate the beginning of the diffuse sound field)
    this is used as the end time of the integration
    t_start is defined as the first point at which the impulse response is not 0
    """

    mask_geo = (t_geo >= t_start) & (t_geo <= t_end)
    mask_wave = (t_wave >= t_start) & (t_wave <= t_end)
    energy_geo = np.trapezoid(ir_geo[mask_geo] ** 2, t_geo[mask_geo])
    energy_wave = np.trapezoid(ir_wav[mask_wave]**2, t_wave[mask_wave])

    eta = np.sqrt(energy_geo / energy_wave)

    return eta


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


def evaluate_rt60_per_band(hybrid_rir, fs=44100):
    from scipy.signal import butter, sosfiltfilt
    
    freq_bands    = [63, 125, 250, 500, 1000, 2000, 4000]
    rt60_sabine   = [1.482, 0.972, 1.110, 1.147, 1.196, 0.877, 0.877]
    rt60_eyring   = [1.430, 0.919, 1.058, 1.095, 1.143, 0.824, 0.824]
    
    print(f"{'Band':>6} {'Hybrid RT60':>12} {'Sabine':>8} {'Eyring':>8} {'vs Eyring':>10}")
    print('-' * 50)
    
    for fc, rt_sab, rt_eyr in zip(freq_bands, rt60_sabine, rt60_eyring):
        fl = fc / np.sqrt(2)
        fh = min(fc * np.sqrt(2), fs/2 * 0.99)
        sos  = butter(4, [fl, fh], btype='band', fs=fs, output='sos')
        band = sosfiltfilt(sos, hybrid_rir)
        sch_band = schroeder_integration(band)
        rt60 = calculate_rt60(sch_band, fs)  # your existing function
        if rt60:
            diff = (rt60/rt_eyr - 1) * 100
            print(f"{fc:>6} {rt60:>12.3f} {rt_sab:>8.3f} {rt_eyr:>8.3f} {diff:>+9.1f}%")
        else:
            print(f"{fc:>6} {'N/A':>12}")

def plot_tf(signals: dict, fs=44100, title="Transfer Function", 
            f_min=20, f_max=500, log_scale=True):
    """
    overlay transfer functions for multiple signals on the same axes.
    signals: dict of {label: array}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, sig in signals.items():
        n   = len(sig)
        f   = np.fft.rfftfreq(n, 1/fs)
        mag = 20 * np.log10(np.abs(np.fft.rfft(sig)) + 1e-12)
        mask = (f >= f_min) & (f <= f_max)
        ax.plot(f[mask], mag[mask], label=label)

    ax.axvline(250, color='red', linestyle='--', linewidth=1, label='Crossover (250 Hz)')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which='both', alpha=0.4)
    if log_scale:
        ax.set_xscale('log')
    plt.tight_layout()
    plt.show()

def create_hybrid(geo_res, wave_res, crossover_hz = 255, fs = 44100):
    """
    full calibration and crossover pipeline
    params: 
    geo_res: returned by run_geometric
    wave_res: returned by run_wave
    crossover_hz: crossover frequency between wave and geo models
    fs: sampling frequency
    """
    #unpack geometric results
    rir_total = geo_res["rir_total"]
    td = geo_res["td"]
    tr = geo_res["tr"]
    amp_direct = geo_res["gd"]
    t_geo = geo_res["t_geo"]

    #unpack wave results
    IR = wave_res["IR_resampled"].copy()
    t_wave = wave_res["t_resampled"]
    IR[t_wave < td] = 0

    amp_direct_scalar = np.mean(np.abs(np.array(amp_direct)))
    rir_g_scaled = scale_Gir(gd = amp_direct_scalar, td = td, rir_total=rir_total)

    geo_lp = low_pass_filter(rir_g_scaled, crossover_hz, fs=fs)
    wave_lp = low_pass_filter(IR, crossover_hz, fs = fs)

    eta = integrate_energy(geo_lp, wave_lp, t_wave, t_geo, t_start=td, t_end=td + 0.03)

    wave_calibrated = IR * eta

    lowband = low_pass_filter(wave_calibrated, crossover_hz, fs=fs, order=4)
    highband = high_pass_filter(rir_g_scaled, fs=fs, min_freq=crossover_hz)

    if len(lowband) != len(highband):
        new_len = max(len(lowband), len(highband))
        print(new_len)
        lowband = np.pad(lowband, (0, new_len - len(lowband)))
        highband = np.pad(highband, (0, new_len - len(highband)))


    hybrid_rir = lowband + highband

    return {
        "hybrid_rir": hybrid_rir,
        "wave_calibrated": wave_calibrated,
        "rir_g_scaled": rir_g_scaled,
        "eta": eta,
        "crossover_hz": crossover_hz,
        "fs": fs,
    }


def compute_metrics(label, rir, td, fs=44100):
    sch  = schroeder_integration(rir)
    rt60 = calculate_rt60(sch, fs=fs)
    edt  = calculate_edt(sch, fs=fs)

    # C80
    n80   = int(0.08 * fs)
    c80   = 10 * np.log10(np.sum(rir[:n80]**2) / (np.sum(rir[n80:]**2) + 1e-12))

    # D50
    n50   = int(0.05 * fs)
    d50   = np.sum(rir[:n50]**2) / (np.sum(rir**2) + 1e-12)

    # Ts
    t     = np.arange(len(rir)) / fs
    ts    = np.sum(t * rir**2) / (np.sum(rir**2) + 1e-12)

    # DRR
    n_td     = int(td * fs)
    n_win    = int(0.0025 * fs)
    direct   = np.sum(rir[max(0, n_td - n_win): n_td + n_win]**2)
    reverb   = np.sum(rir[n_td + n_win:]**2)
    drr      = 10 * np.log10(direct / (reverb + 1e-12))

    print(f"\n  ── {label} ──")
    print(f"  RT60 : {rt60:.3f} s" if rt60 else "  RT60 : N/A")
    print(f"  EDT  : {edt:.3f} s"  if edt  else "  EDT  : N/A")
    print(f"  C80  : {c80:.2f} dB")
    print(f"  D50  : {d50:.3f}")
    print(f"  Ts   : {ts*1000:.1f} ms")
    print(f"  DRR  : {drr:.2f} dB")

    return {"rt60": rt60, "edt": edt, "c80": c80, "d50": d50, "ts": ts, "drr": drr}

        
geo_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\rir_shoebox.npz")
wave_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\processed_data.npz", allow_pickle=True)

rir_total = geo_data["rir_total"]
amp_direct = geo_data["amp_direct"]
td = geo_data["t_d"]
tr = geo_data["t_r"]

wave_rir = wave_data["IR_resampled"]
t_wave = wave_data["t_resampled"]

t_geo = np.linspace(0, 2.0, len(rir_total))


plot_rir(rir_total, fs = 44100)

plt.plot(t_wave, wave_rir)
plt.show()

scaled_geo = scale_Gir(gd = np.mean(np.abs(np.array(amp_direct))), td = td, rir_total = rir_total)

low_pass_geo = low_pass_filter(scaled_geo, cutoff = 300, fs = 44100)
low_pass_wave = low_pass_filter(wave_rir, cutoff = 300, fs = 44100)

eta = calculate_cal_coef_freqdomain(geo_rir = low_pass_geo, wave_rir = low_pass_wave, f_low = 200, f_high = 280)

wave_calibrated = wave_rir * eta

plot_tf({"wave": wave_calibrated, "geo": scaled_geo})
plot_tf({"Geo (raw)": rir_total, "Wave (raw)": wave_rir},
        title="Step 1 — Raw transfer functions")

metrics_geo_raw  = compute_metrics("Geo (raw)",  rir_total, td)
metrics_wave_raw = compute_metrics("Wave (raw)", wave_rir,  td)

lowband, highband = apply_crossover(geo_rir = scaled_geo, wave_rir = wave_calibrated, crossover_hz = 250)

if len(lowband) != len(highband):
    new_len = max(len(lowband), len(highband))
    print(new_len)
    lowband = np.pad(lowband, (0, new_len - len(lowband)))
    highband = np.pad(highband, (0, new_len - len(highband)))


hybrid_rir = lowband + highband

plot_tf({"low band": lowband, "highband": highband, "hybrid": hybrid_rir}, title="crossover and hybrid")

plot_rir(hybrid_rir, fs = 44100)

