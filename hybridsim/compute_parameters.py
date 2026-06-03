"""
Compute room acoustic parameters (RT60, EDT) for multiple RIR datasets
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, spectrogram

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


# ============================================================
# Load data
# ============================================================

geo_data_pos1 = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos1\rir_shoebox_pos1_200000_withphase_ismspec.npz")

geo_left = geo_data_pos1["brir_l"]
geo_right = geo_data_pos1["brir_r"]

geo_data_pos2 = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos3\rir_shoebox_pos3_200000_vectorized.npz")

geo_data_pos3 = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos3\rir_shoebox_pos3_200000_withphase_ismspec.npz")

wave_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\pos2\shoebox_lc05_freq300_2s_avabs_pos2.mat.npz")

hybrid_data = np.load(
    r"C:\Masters\Hybrid\hybridsim\results\pos1\hybrid_rir.npz",
    allow_pickle=True
)["hybrid_rir_left"]
hybrid_data2 = np.load(
    r"C:\Masters\Hybrid\hybridsim\results\pos1\hybrid_rir.npz",
    allow_pickle=True
)["hybrid_rir_right"]

hybrid_mono = np.load(
    r"C:\Masters\Hybrid\hybridsim\results\pos1\hybrid_rir_newcarpet_withphase_ismspec.npz",
    allow_pickle = True
)["arr_0"]

fs, raven_data_pos1 = wavfile.read(
    r"C:\Masters\Hybrid\hybridsim\results\pos1\RAVEN_pos1_RIR.wav"
)

fs, raven_data_pos2 = wavfile.read(
    r"C:\Masters\Hybrid\hybridsim\results\pos2\RAVEN_pos2_RIR.wav"
)

fs, raven_data_pos3 = wavfile.read(
    r"C:\Masters\Hybrid\hybridsim\results\pos3\RAVEN_pos3_RIR.wav"
)


fs, raven_BRIR_pos1 = wavfile.read(
    r"C:\Masters\Hybrid\hybridsim\results\pos1\RAVEN_pos1_BRIR.wav"
)

fs, newraven = wavfile.read(
    r"C:\ITASoftware\Raven\RavenOutput\pepepeidk20260531T222458\ImpulseResponses\2026-05-31\22.24.58\RIR_RT\RIR_RT_PrimarySource0_Receiver0_1_1.wav"
)

plot_rir(newraven, fs = 44100)

left_raven = raven_BRIR_pos1[:, 0]
right_raven = raven_BRIR_pos1[:, 1]

plot_rir(left_raven, fs = 44100)
plot_rir(right_raven, fs = 44100)

print(f"RAVEN mono RIR length  : {len(raven_data_pos1)/fs:.3f}s")
print(f"RAVEN BRIR length      : {raven_BRIR_pos1.shape[0]/fs:.3f}s")
#plot_rir(raven_data, fs = 44100)
# Extract signals
rir_total1 = geo_data_pos1["rir_total"]
rir_bands1 = geo_data_pos1["rir_bands"]

rir_total2 = geo_data_pos2["rir_total"]
rir_bands2 = geo_data_pos2["rir_bands"]

rir_total3 = geo_data_pos3["rir_total"]
rir_bands3 = geo_data_pos3["rir_bands"]

wave_rir = wave_data["IR_resampled"]
wave_times = wave_data["t_resampled"]

td = geo_data_pos1["t_d"]

#low pass raven so that it contains same frequencies as hybrid 
raven_lowpass1 = low_pass_filter(raven_data_pos1, cutoff = 4000, fs = fs)
raven_lowpass2 = low_pass_filter(raven_data_pos2, cutoff = 4000, fs = fs)
raven_lowpass3 = low_pass_filter(raven_data_pos3, cutoff = 4000, fs = fs)

left_raven = low_pass_filter(left_raven, cutoff=4000, fs = 44100)
right_raven = low_pass_filter(right_raven, cutoff=4000, fs = 44100)
# ============================================================
# Visualization
# ============================================================

#plot_rir(rir_total, fs=44100)
plot_rir(wave_rir, fs=44100)


# ============================================================
# Helper functions
# ============================================================

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


# ============================================================
# Broadband RT60 / EDT comparison
# ============================================================


# ============================================================
# Frequency-band RT60 analysis
# ============================================================

rt60_geo_bands1 = []
rt60_geo_bands2 = []
rt60_geo_bands3 = []
rt60_wave_bands = []
rt60_raven_bands1 = []
rt60_raven_bands2 = []
rt60_raven_bands3 = []

rt60_raven_bands1_left = []
rt60_raven_bands1_right = []

rt60_hybrid_bands = []
rt60_hybrid_bands_avabs=[]

for f in FREQ_BANDS:

    # Wave model band RT60
    rt60_wave_bands.append(
        compute_rt60_from_rir(bandpass_rir(wave_rir, f), td)
    )

    # Raven RIR band RT60
    rt60_raven_bands1.append(
        compute_rt60_from_rir(bandpass_rir(raven_data_pos1, f), td)
    )

    rt60_raven_bands2.append(
        compute_rt60_from_rir(bandpass_rir(raven_data_pos2, f), td)
    )

    rt60_raven_bands3.append(
        compute_rt60_from_rir(bandpass_rir(raven_data_pos3, f), td)
    )

    rt60_raven_bands1_left.append(
        compute_rt60_from_rir(bandpass_rir(left_raven, f), td)
    )

    rt60_raven_bands1_right.append(
        compute_rt60_from_rir(bandpass_rir(right_raven, f), td)
    )

    # Hybrid RIR band RT60
    rt60_hybrid_bands.append(
        compute_rt60_from_rir(bandpass_rir(hybrid_data, f), td)
    )

    rt60_hybrid_bands_avabs.append(
        compute_rt60_from_rir(bandpass_rir(hybrid_data2, f), td)
    )

for band in rir_bands1:
    rt60_geo_bands1.append(
        compute_rt60_from_rir(band, td)
    )

for band in rir_bands2:
    rt60_geo_bands2.append(
        compute_rt60_from_rir(band, td)
    )

for band in rir_bands3:
    rt60_geo_bands3.append(
        compute_rt60_from_rir(band, td)
    )
# ============================================================
# Results
# ============================================================


print("\nRT60 (Geo) pos 1:")
print(rt60_geo_bands1)



average_RAVEN = np.mean(np.stack([rt60_raven_bands1, rt60_raven_bands2, rt60_raven_bands3], axis=0), axis=0)
average_GEO = np.mean(np.stack([rt60_geo_bands1, rt60_geo_bands2, rt60_geo_bands3], axis=0), axis=0)

print("\nGEO averaged across all pos:")
print(average_GEO)

print("\nRAVEN averaged across all pos:")
print(average_RAVEN)

print("\nRaven broadband: ")
print(np.mean([compute_rt60_from_rir(raven_lowpass1, td), compute_rt60_from_rir(raven_lowpass2, td), compute_rt60_from_rir(raven_lowpass3, td)]))

print("\nGEO broadband")
print(np.mean([compute_rt60_from_rir(rir_total1, td), compute_rt60_from_rir(rir_total2, td), compute_rt60_from_rir(rir_total3, td)]))

print("\nHybrid left : ")
print(rt60_hybrid_bands)
print("\nRaven left:")
print(rt60_raven_bands1_left)
print("\nHybrid right: ")
print(rt60_hybrid_bands_avabs)
print("\nRaven right:")
print(rt60_raven_bands1_right)

import numpy as np

# Room volume
V = 5 * 4 * 3  # 60 m^3

# Surface areas
S_floor = 20
S_ceiling = 20
S_walls = 54

# Absorption coefficients
alpha_walls = np.array([0.08, 0.11, 0.05, 0.03, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03])
alpha_floor = np.array([0.03, 0.09, 0.25, 0.31, 0.33, 0.44, 0.44, 0.44, 0.44, 0.44])

# Total absorption per band
A = (S_floor * alpha_floor +
     (S_ceiling + S_walls) * alpha_walls)

# Sabine RT60 per band
RT60 = 0.161 * V / A

print("RT60 per band (Sabine):")
print(RT60)

t_raven, sch_raven = schroeder_decay(raven_lowpass1)
t_geo, sch_geo = schroeder_decay(rir_total1)
t_wav, sch_wav = schroeder_decay(wave_rir)
t_hybrid, sch_hybrid = schroeder_decay(hybrid_data)

fig, ax = plt.subplots()
ax.plot(t_raven, sch_raven, label = "raven")
ax.plot(t_geo, sch_geo, label = "geo")
ax.plot(t_wav, sch_wav, label = "wave")
ax.plot(t_hybrid, sch_hybrid, label = "hybrid")
ax.set_ylim(-60)
plt.title("schroeder decay curves")
plt.legend()
plt.show()


fig1, ax1 = plt.subplots()
ax1.plot(FREQ_BANDS, rt60_geo_bands1, label = "geo")
ax1.plot(FREQ_BANDS, rt60_hybrid_bands, label = "hybrid")
ax1.plot(FREQ_BANDS, rt60_raven_bands1, label = "raven")
ax1.plot(FREQ_BANDS, RT60[:7], label = "sabine")
plt.title("per-band RT60")
plt.legend()
plt.show()

fig1, ax1 = plt.subplots()
ax1.plot(FREQ_BANDS, rt60_hybrid_bands, label = "hybrid, freqdep carpet")
ax1.plot(FREQ_BANDS, rt60_hybrid_bands_avabs, label = "hybrid, avabs")
ax1.plot(FREQ_BANDS, rt60_raven_bands1, label = "raven")
ax1.plot(FREQ_BANDS, RT60[:7], label = "sabine")
plt.title("comparing BCs")
plt.legend()
plt.show()

print(f"RAVEN broadband left RT60: {compute_edt_from_rir(left_raven)}")
print(f"Raven mono broadband: {compute_edt_from_rir(raven_lowpass1)}")
print(f"geo broadband left: {compute_edt_from_rir(geo_left)}")
print(f"geo broadband right: {compute_edt_from_rir(geo_right)}")
print(f"Raven right broadband: {compute_edt_from_rir(right_raven)}")
print(f"hybrid mono broadband: {compute_edt_from_rir(hybrid_mono)}")
print(f"hybrid left BRIR RT60: {compute_edt_from_rir(hybrid_data)}")
print(f"hybrid right BRIR RT60: {compute_edt_from_rir(hybrid_data2)}")

