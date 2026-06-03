
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
    print(f"eta = max_geo/max_wave = {max_geo/max_wave:.6f}")

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

def align_rirs(geo_rir, wave_rir, t_geo, t_wave, td, tr, fs=44100):

    search_start = td * 0.5
    search_end   = td + (tr - td)

    mask_wave          = (t_wave >= search_start) & (t_wave <= search_end)
    window_values      = np.abs(wave_rir[mask_wave])
    window_times       = t_wave[mask_wave]

    if window_values.size == 0:
        raise ValueError(f"No samples in search window "
                         f"[{search_start*1000:.2f}, {search_end*1000:.2f}]ms")

    peak_idx_in_window = np.argmax(window_values)
    t_wave_peak        = window_times[peak_idx_in_window]

    # shift in samples -- this is the corrected calculation
    shift_seconds = td - t_wave_peak
    shift_samples = int(np.round(shift_seconds * fs))

    print(f"  td (geo)           : {td*1000:.3f} ms")
    print(f"  Wave peak at       : {t_wave_peak*1000:.3f} ms")
    print(f"  Shift              : {shift_seconds*1000:.3f} ms = {shift_samples} samples")

    # apply shift
    if shift_samples > 0:
        wave_aligned = np.pad(wave_rir, (shift_samples, 0))[:len(wave_rir)]
    elif shift_samples < 0:
        wave_aligned = np.pad(wave_rir[abs(shift_samples):],
                              (0, abs(shift_samples)))
    else:
        wave_aligned = wave_rir.copy()

    # ── validation ────────────────────────────────────────────────────────
    # recheck peak position after shift using global sample index
    global_indices  = np.where(mask_wave)[0]
    shifted_indices = global_indices + shift_samples
    shifted_indices = shifted_indices[(shifted_indices >= 0) &
                                      (shifted_indices < len(wave_aligned))]

    if len(shifted_indices) > 0:
        peak_after      = np.argmax(np.abs(wave_aligned[shifted_indices]))
        t_peak_after    = shifted_indices[peak_after] / fs
        residual_ms     = abs(t_peak_after - td) * 1000
        residual_samps  = int(np.round(residual_ms * fs / 1000))
        print(f"  Peak after shift   : {t_peak_after*1000:.3f} ms")
        print(f"  Residual error     : {residual_ms:.3f} ms ({residual_samps} samples)")

        if residual_samps > 1:
            print(f"  ✗ WARNING: residual > 1 sample -- likely speed of sound mismatch")
            print(f"    check c_sound is identical in both simulations")
        else:
            print(f"  ✓ alignment within 1 sample")

    # ── plot with matched amplitudes ──────────────────────────────────────
    # normalise both to their own peak for visual comparison
    geo_norm  = geo_rir  / (np.max(np.abs(geo_rir))  + 1e-12)
    wav_norm  = wave_rir / (np.max(np.abs(wave_rir))  + 1e-12)
    wav_norm_aligned = wave_aligned / (np.max(np.abs(wave_aligned)) + 1e-12)

    n_plot = int(0.05 * fs)
    t_ms   = np.arange(n_plot) / fs * 1000

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(t_geo[:n_plot]  * 1000, geo_norm[:n_plot],
                 label="Geo (normalised)", color='steelblue')
    axes[0].plot(t_wave[:n_plot] * 1000, wav_norm[:n_plot],
                 label="Wave (normalised, before)", color='coral', alpha=0.8)
    axes[0].axvline(td * 1000, color='green', linestyle='--',
                    label=f'td={td*1000:.2f}ms')
    axes[0].set_title("Before alignment (normalised amplitudes)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].plot(t_geo[:n_plot]  * 1000, geo_norm[:n_plot],
                 label="Geo (normalised)", color='steelblue')
    axes[1].plot(t_wave[:n_plot] * 1000, wav_norm_aligned[:n_plot],
                 label="Wave (normalised, aligned)", color='green', alpha=0.8)
    axes[1].axvline(td * 1000, color='green', linestyle='--',
                    label=f'td={td*1000:.2f}ms')
    axes[1].set_title("After alignment (normalised amplitudes)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    axes[1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()

    return wave_aligned, shift_samples


def apply_crossover(geo_rir, wave_rir, crossover_hz, fs=44100):
    from scipy.signal import butter, sosfiltfilt
    
    # 2nd order Butterworth — sosfiltfilt makes it effectively 4th order (LR-4)
    sos_low  = butter(4, crossover_hz, btype='low',  fs=fs, output='sos')
    sos_high = butter(4, crossover_hz, btype='high', fs=fs, output='sos')


    low_band  = sosfiltfilt(sos_low,  wave_rir)
    high_band = sosfiltfilt(sos_high, geo_rir)
    
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
            f_min=20, f_max=800, log_scale=False):
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
    ax.set_xlim(0,800)
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
    brir_l = geo_res["brir_l"]
    brir_r = geo_res["brir_r"]
    td = geo_res["td"]
    tr = geo_res["tr"]
    amp_direct = geo_res["gd"]
    t_geo = geo_res["t_geo"]

    #unpack wave results
    IR = wave_res["IR_resampled"].copy()
    t_wave = wave_res["t_resampled"]
    IR[t_wave < td] = 0

    amp_direct_scalar = np.mean(np.abs(np.array(amp_direct)))
    rir_left_scaled = scale_Gir(gd = amp_direct_scalar, td = td, rir_total=brir_l)
    rir_right_scaled = scale_Gir(gd = amp_direct_scalar, td = td, rir_total=brir_r)

    geo_left_lp = low_pass_filter(rir_left_scaled, crossover_hz, fs=fs)
    geo_right_lp = low_pass_filter(rir_right_scaled, crossover_hz, fs = fs)
    wave_lp = low_pass_filter(IR, crossover_hz, fs = fs)

    eta_left = integrate_energy(geo_left_lp, wave_lp, t_wave, t_geo, t_start=td, t_end=td + 0.03)
    eta_right = integrate_energy(geo_right_lp, wave_lp, t_wave, t_geo, t_start=td, t_end = td + 0.03)
    eta_av = np.mean([eta_left, eta_right])
    wave_calibrated_left = IR * eta_av
    wave_calibrated_right = IR * eta_av

    lowband_left = low_pass_filter(wave_calibrated_left, crossover_hz, fs=fs, order=4)
    highband_left = high_pass_filter(rir_left_scaled, fs=fs, min_freq=crossover_hz)

    lowband_right = low_pass_filter(wave_calibrated_right, crossover_hz, fs=fs, order=4)
    highband_right = high_pass_filter(rir_right_scaled, fs=fs, min_freq=crossover_hz)

    if len(lowband_left) != len(highband_left):
        new_len = max(len(lowband_left), len(highband_left))
        lowband_left = np.pad(lowband_left, (0, new_len - len(lowband_left)))
        highband_left = np.pad(highband_left, (0, new_len - len(highband_left)))
        lowband_right = np.pad(lowband_right, (0, new_len - len(lowband_right)))
        highband_right = np.pad(highband_right, (0, new_len - len(highband_right)))       


    hybrid_rir_left = lowband_left + highband_left
    hybrid_rir_right = lowband_right + highband_right

    return {
        "hybrid_rir_left": hybrid_rir_left,
        "hybrid_rir_right": hybrid_rir_right,
        "wave_calibrated": wave_calibrated,
        "rir_g_scaled_left": rir_left_scaled,
        "rir_g_scaled_right": rir_right_scaled,
        "eta_left": eta_left,
        "eta_right": eta_right,
        "eta_av": eta_av,
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

def main():
    geo_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos1\rir_shoebox_pos1_200000_withphase_ismspec.npz")
    wave_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos1\shoebox_lc05_freq300_2s_avabs_newcarpet_pos1.mat.npz")

    brir_left = geo_data["brir_l"]
    brir_right = geo_data["brir_r"]
    rir_total = geo_data["rir_total"]

    amp_direct = geo_data["amp_direct"]
    td = geo_data["t_d"]
    tr = geo_data["t_r"]

    wave_rir = wave_data["IR_resampled"]
    t_wave = wave_data["t_resampled"]

    t_geo = np.linspace(0, 2.0, len(rir_total))

    plot_rir(brir_left, fs = 44100)
    plot_rir(brir_right, fs = 44100)

    scaled_geo_left = scale_Gir(gd = np.mean(np.abs(np.array(amp_direct))), td = td, rir_total = brir_left)
    scaled_geo_right = scale_Gir(gd = np.mean(np.abs(np.array(amp_direct))), td = td, rir_total = brir_right)


    low_pass_geo_left = low_pass_filter(scaled_geo_left, cutoff = 300, fs = 44100)
    low_pass_geo_right = low_pass_filter(scaled_geo_right, cutoff = 300, fs = 44100)
    low_pass_wave = low_pass_filter(wave_rir, cutoff = 300, fs = 44100)

    eta_left = integrate_energy(ir_geo = low_pass_geo_left, ir_wav = low_pass_wave, t_wave = t_wave, t_geo = t_geo, t_start = td, t_end = 0.02)
    eta_right = integrate_energy(ir_geo = low_pass_geo_right, ir_wav = low_pass_wave, t_wave = t_wave, t_geo = t_geo, t_start = td, t_end = 0.02)
    eta_av = (eta_left + eta_right) / 2
    print(eta_left, eta_right, eta_av)
    #eta1 = calculate_cal_coef_freqdomain(geo_rir = low_pass_geo, wave_rir = low_pass_wave, f_low = 200, f_high = 280)
    #eta2 = integrate_energy(ir_geo = low_pass_geo, ir_wav = low_pass_wave, t_wave = t_wave, t_geo = t_geo, t_start = td, t_end = 0.03)
    #print(f"eta2: {eta2}")
    wave_calibrated = wave_rir * eta_av

    plot_tf({"wave": wave_calibrated, "geo": scaled_geo_left}, title="raw transfer functions, left ear")
    plot_tf({"wave": wave_calibrated, "geo": scaled_geo_right}, title="raw transfer functions, right ear")

    #plot_tf({"Geo (raw)": rir_total, "Wave (raw)": wave_rir},
            #title="Step 1 — Raw transfer functions")

    #metrics_geo_raw  = compute_metrics("Geo (raw)",  rir_total, td)
    #metrics_wave_raw = compute_metrics("Wave (raw)", wave_rir,  td)

    lowband_left, highband_left = apply_crossover(geo_rir = scaled_geo_left, wave_rir = wave_calibrated, crossover_hz = 250)
    lowband_right, highband_right = apply_crossover(geo_rir = scaled_geo_right, wave_rir = wave_calibrated, crossover_hz = 250)


    if len(lowband_left) != len(highband_left):
        new_len = max(len(lowband_left), len(highband_left))
        print(new_len)
        lowband_left = np.pad(lowband_left, (0, new_len - len(lowband_left)))
        highband_left = np.pad(highband_left, (0, new_len - len(highband_left)))

    if len(lowband_right) != len(highband_right):
        new_len = max(len(lowband_right), len(highband_right))
        print(new_len)
        lowband_right = np.pad(lowband_right, (0, new_len - len(lowband_right)))
        highband_right = np.pad(highband_right, (0, new_len - len(highband_right)))


    hybrid_rir_left = lowband_left + highband_left
    hybrid_rir_right = lowband_right + highband_right


    np.savez(r"C:\Masters\Hybrid\hybridsim\results\hybrid_rir.npz", hybrid_rir_left = hybrid_rir_left, hybrid_rir_right = hybrid_rir_right)

    plot_tf({"low band": lowband_left, "highband": highband_left, "hybrid": hybrid_rir_left}, title="crossover and hybrid,left ear")
    plot_tf({"low band": lowband_right, "highband": highband_right, "hybrid": hybrid_rir_right}, title="crossover and hybrid,right ear")


    plot_rir(hybrid_rir_left, fs = 44100)
    plot_rir(hybrid_rir_right, fs = 44100)
"""
def main():
    geo_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos1\rir_shoebox_pos1_200000_withphase_ismspec.npz")
    wave_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos1\shoebox_lc05_freq300_2s_avabs_newcarpet_pos1.mat.npz")
    rir_total = geo_data["rir_total"]
    brir_left = geo_data["brir_l"]
    brir_right = geo_data["brir_r"]

    amp_direct = geo_data["amp_direct"]
    td = geo_data["t_d"]
    tr = geo_data["t_r"]

    print(td)

    wave_rir = wave_data["IR_resampled"]
    t_wave = wave_data["t_resampled"]

    t_geo = np.linspace(0, 2.0, len(rir_total))

    plot_rir(rir_total, fs = 44100)
    plt.plot(t_wave, wave_rir)

    IR, shift_samples = align_rirs(rir_total, wave_rir, t_geo, t_wave, td, tr)
    n_td = int(td * 44100)
    IR[:n_td] = 0
    wave_rir = IR

    scaled_geo = scale_Gir(gd = np.mean(np.abs(np.array(amp_direct))), td=td, rir_total=rir_total)
    #scaled_geo = rir_total
    low_pass_geo = low_pass_filter(rir=scaled_geo, cutoff=250, fs=44100)
    low_pass_wave = low_pass_filter(rir=wave_rir, cutoff=250, fs = 44100)
    plot_rir(low_pass_geo, fs =44100)
    eta = integrate_energy(ir_geo = low_pass_geo, ir_wav = low_pass_wave, t_wave = t_wave, t_geo = t_geo, t_start = td, t_end = td + 0.02)
    #eta = calculate_cal_coef_freqdomain(geo_rir = low_pass_geo, wave_rir = low_pass_wave, fs = 44100, f_low =200, f_high = 240)
    #eta = calculate_cal_coef(td=td, tr = tr, geo_rir= low_pass_geo, wave_rir=low_pass_wave, t_wave=t_wave, t_geo= t_geo)
    print(eta)
    wave_scaled = wave_rir * eta
    plot_tf({"wave": wave_scaled, "geo": scaled_geo}, title="calibrated")

    plot_tf({"wave": wave_rir, "geo": scaled_geo}, title="raw transfer f")

    lowband, highband = apply_crossover(geo_rir = scaled_geo, wave_rir = wave_scaled, crossover_hz = 250)


    if len(lowband) != len(highband):
        new_len = max(len(lowband), len(highband))
        print(new_len)
        lowband = np.pad(lowband, (0, new_len - len(lowband)))
        highband = np.pad(highband, (0, new_len - len(highband)))

    n = min(len(scaled_geo), len(wave_scaled))
    f = np.fft.rfftfreq(n, 1/44100)
    F_geo  = np.fft.rfft(scaled_geo[:n])
    F_wave = np.fft.rfft(wave_scaled[:n])

    # Find index closest to crossover
    idx = np.argmin(np.abs(f - 270))
    print(f"Geo phase at 250Hz:  {np.angle(F_geo[idx],  deg=True):.1f}°")
    print(f"Wave phase at 250Hz: {np.angle(F_wave[idx], deg=True):.1f}°")


    hybrid_rir = lowband + highband
    plot_tf({"lowband": lowband, "highband": highband, "hybrid": hybrid_rir}, title='crossover')

    plot_tf({"hybrid": hybrid_rir}, title = "hybrid RIR")

    plot_rir(hybrid_rir, fs = 44100)
    np.savez(r"C:\Masters\Hybrid\hybridsim\results\pos1\hybrid_rir_newcarpet_withphase_ismspec.npz", hybrid_rir)




    plot_rir(rir_total, fs = 44100)

    plt.plot(t_wave, wave_rir)
    plt.show()

    scaled_geo_left = scale_Gir(gd = np.mean(np.abs(np.array(amp_direct))), td = td, rir_total = brir_left)
    scaled_geo_right = scale_Gir(gd = np.mean(np.abs(np.array(amp_direct))), td = td, rir_total = brir_right)


    low_pass_geo_left = low_pass_filter(scaled_geo_left, cutoff = 300, fs = 44100)
    low_pass_geo_right = low_pass_filter(scaled_geo_right, cutoff = 300, fs = 44100)
    low_pass_wave = low_pass_filter(wave_rir, cutoff = 300, fs = 44100)

    eta_left = integrate_energy(ir_geo = low_pass_geo_left, ir_wav = low_pass_wave, t_wave = t_wave, t_geo = t_geo, t_start = td, t_end = 0.03)
    eta_right = integrate_energy(ir_geo = low_pass_geo_right, ir_wav = low_pass_wave, t_wave = t_wave, t_geo = t_geo, t_start = td, t_end = 0.03)
    eta_av = (eta_left + eta_right) / 2
    print(eta_left, eta_right, eta_av)
    #eta1 = calculate_cal_coef_freqdomain(geo_rir = low_pass_geo, wave_rir = low_pass_wave, f_low = 200, f_high = 280)
    #eta2 = integrate_energy(ir_geo = low_pass_geo, ir_wav = low_pass_wave, t_wave = t_wave, t_geo = t_geo, t_start = td, t_end = 0.03)
    #print(f"eta2: {eta2}")
    wave_calibrated = wave_rir * eta_av

    plot_tf({"wave": wave_calibrated, "geo": scaled_geo_left}, title="raw transfer functions, left ear")
    plot_tf({"wave": wave_calibrated, "geo": scaled_geo_right}, title="raw transfer functions, right ear")

    #plot_tf({"Geo (raw)": rir_total, "Wave (raw)": wave_rir},
            #title="Step 1 — Raw transfer functions")

    #metrics_geo_raw  = compute_metrics("Geo (raw)",  rir_total, td)
    #metrics_wave_raw = compute_metrics("Wave (raw)", wave_rir,  td)

    lowband_left, highband_left = apply_crossover(geo_rir = scaled_geo_left, wave_rir = wave_calibrated, crossover_hz = 250)
    lowband_right, highband_right = apply_crossover(geo_rir = scaled_geo_right, wave_rir = wave_calibrated, crossover_hz = 250)


    if len(lowband_left) != len(highband_left):
        new_len = max(len(lowband_left), len(highband_left))
        print(new_len)
        lowband_left = np.pad(lowband_left, (0, new_len - len(lowband_left)))
        highband_left = np.pad(highband_left, (0, new_len - len(highband_left)))

    if len(lowband_right) != len(highband_right):
        new_len = max(len(lowband_right), len(highband_right))
        print(new_len)
        lowband_right = np.pad(lowband_right, (0, new_len - len(lowband_right)))
        highband_right = np.pad(highband_right, (0, new_len - len(highband_right)))


    hybrid_rir_left = lowband_left + highband_left
    hybrid_rir_right = lowband_right + highband_right


    np.savez(r"C:\Masters\Hybrid\hybridsim\results\hybrid_rir.npz", hybrid_rir_left, hybrid_rir_right)

    plot_tf({"low band": lowband_left, "highband": highband_left, "hybrid": hybrid_rir_left}, title="crossover and hybrid,left ear")
    plot_tf({"low band": lowband_right, "highband": highband_right, "hybrid": hybrid_rir_right}, title="crossover and hybrid,right ear")


    plot_rir(hybrid_rir_left, fs = 44100)
    plot_rir(hybrid_rir_right, fs = 44100)
    """

if __name__ == "__main__":
    main()