from scipy.io import loadmat
import numpy as np
from rayroom.core.data_anal import plot_rir
from rayroom.analytics.acoustics import (
    schroeder_integration,
    calculate_rt60,
    calculate_edt,
    schroeder_decay,
)
from rayroom.core.constants import FREQ_BANDS
from rayroom.core.data_anal import plot_rir
import matplotlib.pyplot as plt
from calibration import low_pass_filter
from scipy.signal import butter, sosfilt, spectrogram



def compute_rt60_from_rir(rir, td, fs=44100):
    """Compute RT60 using Schroeder integration."""
    n_td = int(td * fs)
    band_from_direct = rir[n_td:]
    sch = schroeder_integration(band_from_direct)
    return calculate_rt60(sch, fs=fs)


def compute_edt_from_rir(rir, fs=44100):
    """Compute EDT using Schroeder integration."""
    sch = schroeder_integration(rir)
    return calculate_edt(sch, fs=fs)

def bandpass_rir(rir, center_freq, fs=44100, order=6):
    """
    Bandpass filter RIR around 1/3 octave band centered at `center_freq`
    """
    f_low = center_freq / (2 ** (1 / 6))
    f_high = center_freq * (2 ** (1 / 6))

    nyq = fs / 2
    sos = butter(
        order,
        [f_low / nyq, f_high / nyq],
        btype="band",
        output="sos"
    )
    return sosfilt(sos, rir)

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


    hybrid_ref = irs[2,0]

    hybrid_ref_sc2_rec2 = hybrid_ref[1]
    print(hybrid_ref_sc2_rec2.shape)

    TD_DG_sc2_rec2 = TD_DG[1]
    print(TD_DG_sc2_rec2.shape)

    geo_sc1_rec2 = np.load(r"C:\Masters\Hybrid\hybridsim\scenario1\results\rir_scenario1_ism5_200000_S2R2_wallmat.npz")
    hybrid = np.load(r"C:\Masters\Hybrid\hybridsim\scenario1\results\hybrid_rir_wallmat.npz")["hybrid_mono"]

    rir = geo_sc1_rec2["rir_total"]
    rir_bands = geo_sc1_rec2["rir_bands"]
    td = geo_sc1_rec2["t_d"]

    geo_sc1_rt60 = []
    hybrid_rt60 = []
    hybrid_ref_rt60 = []
    TD_DG_rt60 = []

    geo_EDT = []
    hybrid_EDT = []
    hybrid_ref_EDT = []
    TD_DG_EDT = []


    for band in rir_bands:
        geo_sc1_rt60.append(compute_rt60_from_rir(band, td = td))
        geo_EDT.append(compute_edt_from_rir(band))


    for f in FREQ_BANDS:
        hybrid_rt60.append(compute_rt60_from_rir(bandpass_rir(hybrid, center_freq=f), td = td, fs = 44100))
        hybrid_ref_rt60.append(compute_rt60_from_rir(bandpass_rir(hybrid_ref_sc2_rec2, center_freq=f), td = td, fs = 44800))
        TD_DG_rt60.append(compute_rt60_from_rir(bandpass_rir(TD_DG_sc2_rec2, center_freq=f), td=td, fs = 44800))
        #geo_sc1_rt60.append(compute_rt60_from_rir(bandpass_rir(rir, center_freq=f), td=td, fs= 44100))
        hybrid_EDT.append(compute_edt_from_rir(bandpass_rir(hybrid, center_freq=f)))
        hybrid_ref_EDT.append(compute_edt_from_rir(bandpass_rir(hybrid_ref_sc2_rec2, center_freq=f), fs = 44800))
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


if __name__ == "__main__":
    main()
                           


  
