from scipy.io import loadmat
import numpy as np
from rayroom.core.data_anal import plot_rir
from rayroom.analytics.acoustics import (
    schroeder_integration,
    calculate_rt60,
    calculate_edt,
    schroeder_decay,
    octave_band_filter
)
from rayroom.core.constants import FREQ_BANDS
from rayroom.core.data_anal import plot_rir
import matplotlib.pyplot as plt
from calibration import low_pass_filter, compute_metrics
from scipy.signal import butter, sosfilt, spectrogram



def compute_rt60_from_rir(rir, fs=44100):
    """Compute RT60 using Schroeder integration."""
    sch = schroeder_integration(rir)
    return calculate_rt60(sch, fs=fs)


def compute_edt_from_rir(rir, fs=44100):
    """Compute EDT using Schroeder integration."""
    sch = schroeder_integration(rir)
    return calculate_edt(sch, fs=fs)

def bandpass_rir(rir, center_freq, fs=44100, order=6):
    """
    Bandpass filter RIR around 1/3 octave band centered at `center_freq`
    """
    f_low = center_freq / (2 ** (1/2))
    f_high = center_freq * (2 ** (1 / 2))

    nyq = fs / 2
    sos = butter(
        order,
        [f_low / nyq, f_high / nyq],
        btype="band",
        output="sos"
    )
    return sosfilt(sos, rir)

def bandpass_paper_range(signal, fs=44100):

    from scipy.signal import sosfiltfilt
    """Filter to paper's 125-2000Hz range"""
    sos = butter(8, [125, 2000], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)


def main():
    hybrid_val = loadmat(r"C:\Masters\room_impulse_reponses_hybrid_model_paper.mat")

    sc1 = hybrid_val["Irs"]["scenario_1"][0, 0]
    fs = hybrid_val["Irs"]["fs"]
    print(fs)
    inner = sc1[0, 0]


    # It's a struct with field 'irs'
    irs = inner['irs']
    print(f"irs shape: {irs.shape}, dtype: {irs.dtype}")

    TD_DG = irs[0,0]


    hybrid_ref = irs[1,0]

    hybrid_ref_sc2_rec2 = hybrid_ref[11]
    print(hybrid_ref_sc2_rec2.shape)

    TD_DG_sc2_rec2 = TD_DG[11]
    print(TD_DG_sc2_rec2.shape)

    #geo_sc1_rec2 = np.load(r"C:\Masters\Hybrid\hybridsim\scenario1\results\rir_scenario1_ism5_200000_S2R2_wallmat.npz")
    hybrid_data = np.load(r"C:\Masters\Hybrid\hybridsim\scenario1\results\hybrid_pipeline_S2R2_newcalibration.npz")
    #hybrid = hybrid_data["hybrid_mono"]
    hybrid_data1 = np.load(r"C:\Masters\Hybrid\hybridsim\scenario1\results\hybrid_sc1_result_pipeline_S2R12")
    hybrid = hybrid_data1["hybrid_mono"]
    rir = hybrid_data1["rir_total"]
    rir_bands = hybrid_data1["rir_bands"]
    td = hybrid_data1["td"]
    print(hybrid_data1["geo_duration"])
    print(hybrid_data1["wave_duration"])

    geo_sc1_rt60 = []
    hybrid_rt60 = []
    hybrid_ref_rt60 = []
    TD_DG_rt60 = []

    geo_EDT = []
    hybrid_EDT = []
    hybrid_ref_EDT = []
    TD_DG_EDT = []


    for band in rir_bands:
        geo_sc1_rt60.append(compute_rt60_from_rir(band))
        geo_EDT.append(compute_edt_from_rir(band))


    for f in FREQ_BANDS:
        hybrid_rt60.append(compute_rt60_from_rir(bandpass_rir(hybrid, center_freq=f), fs = 44100))
        hybrid_ref_rt60.append(compute_rt60_from_rir(bandpass_rir(hybrid_ref_sc2_rec2, center_freq=f), fs = 48000))
        TD_DG_rt60.append(compute_rt60_from_rir(bandpass_rir(TD_DG_sc2_rec2, center_freq=f), fs = 48000))
        #geo_sc1_rt60.append(compute_rt60_from_rir(bandpass_rir(rir, center_freq=f), td=td, fs= 44100))
        hybrid_EDT.append(compute_edt_from_rir(bandpass_rir(hybrid, center_freq=f), fs = 44100))
        hybrid_ref_EDT.append(compute_edt_from_rir(bandpass_rir(hybrid_ref_sc2_rec2, center_freq=f), fs = 48000))
        TD_DG_EDT.append(compute_edt_from_rir(bandpass_rir(TD_DG_sc2_rec2, center_freq=f), fs = 48000))
                          

    print(geo_sc1_rt60)
    print(hybrid_ref_rt60)
    print(TD_DG_rt60)
    print(hybrid_rt60)

    fig, ax = plt.subplots()
    ax.plot(FREQ_BANDS[:6], geo_sc1_rt60[:6], label = "geo")
    ax.plot(FREQ_BANDS[:6], hybrid_ref_rt60[:6], label = "hybrid ref")
    ax.plot(FREQ_BANDS[:6], TD_DG_rt60[:6], label = "wave")
    ax.plot(FREQ_BANDS[:6], hybrid_rt60[:6], label = "real hybrid")
    ax.set_ylabel("RT60")
    ax.set_xlabel("frequency")
    ax.set_title("RT60 Hybrid validation")
    plt.legend()
    plt.savefig('rt60 hybrid comparison.png', dpi=150)
    plt.show()

    ig, ax = plt.subplots()
    ax.plot(FREQ_BANDS[:6], geo_EDT[:6], label = "geo")
    ax.plot(FREQ_BANDS[:6], hybrid_ref_EDT[:6], label = "hybrid ref")
    ax.plot(FREQ_BANDS[:6], TD_DG_EDT[:6], label = "wave")
    ax.plot(FREQ_BANDS[:6], hybrid_EDT[:6], label = "real hybrid")
    ax.set_ylabel("EDT")
    ax.set_xlabel("frequency")
    ax.set_title("EDT Hybrid validation")
    plt.legend()
    plt.savefig('EDT hybrid comparison.png', dpi=150)
    plt.show()
    hybrid_lowpass = low_pass_filter(rir = hybrid, cutoff = 2000, fs = 44100)
    compute_metrics(label="hybrid reference", rir=hybrid_ref_sc2_rec2, td = td, fs = 44800)
    compute_metrics(label="hybrid maya", rir = hybrid_lowpass, td=td, fs = 44100)

    # 1. Check actual duration
    print(f"TD_DG duration: {len(TD_DG_sc2_rec2)/44800:.3f} s")

    # 2. Plot the Schroeder curves side by side
    sch_dg  = schroeder_integration(TD_DG_sc2_rec2)
    sch_maya = schroeder_integration(hybrid)
    sch_ref = schroeder_integration(hybrid_ref_sc2_rec2)

    plt.plot(sch_dg,  label="TD_DG")
    plt.plot(sch_ref, label="hybrid_ref")
    plt.plot(sch_maya, label="maya")
    plt.ylabel("dB"); plt.legend(); plt.show()

    # 3. Check the fs value from the mat file
    print(f"fs from mat: {fs}")
    # Maximum reliable frequency for DG solver
    # rule of thumb: ~6 DOF per wavelength for P=3 polynomial order
    c = 343  # m/s
    h = 0.5# your typical mesh element size in metres
    P = 4# your polynomial order
    f_max = c * P / (6 * h)
    print(f"Reliable up to: {f_max:.0f} Hz")

if __name__ == "__main__":
    main()
                           


  
