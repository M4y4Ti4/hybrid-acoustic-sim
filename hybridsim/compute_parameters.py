"""
computes room acoustic parameters for the RIRs
"""
import numpy as np
from rayroom.analytics.acoustics import schroeder_integration, calculate_rt60, calculate_edt, _calculate_decay_time, calculate_drr
from rayroom.core.constants import FREQ_BANDS
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG
"""

geo_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\rir_shoebox.npz")
wave_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\processed_data.npz", allow_pickle=True)


wave_rir = wave_data["IR_resampled"]
wave_times = wave_data["t_resampled"]
rir_total = geo_data["rir_total"]
rir_bands = geo_data["rir_bands"]
EDT_bands = []

plot_rir(rir_total, fs = 44100)
plot_rir(wave_rir, fs = 44100)

print(rir_bands.shape)

RT60_bands = []
for b in range(len(FREQ_BANDS)): 
    sch_db_band = schroeder_integration(rir_bands[b,:])
    RT60_bands.append(calculate_rt60(sch_db_band, fs = 44100))

print(RT60_bands)


#calculating RT60 for wave and geo model
sch_db_wave = schroeder_integration(wave_rir)
sch_db_geo = schroeder_integration(rir_total)

RT60_wave = calculate_rt60(sch_db_wave, fs = 44100)
RT60_geo = calculate_rt60(sch_db_geo, fs = 44100)

#calculating EDT (early decay time)
EDT_wave = calculate_edt(sch_db_wave, fs = 44100)
EDT_geo = calculate_edt(sch_db_geo, fs = 44100)

print(RT60_geo, RT60_wave)
print(EDT_geo,EDT_wave)


"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

fs, rir = wavfile.read(r"C:\ITASoftware\Raven\RavenOutput\untitled20260428T111023\ImpulseResponses\2026-04-28\11.10.23\RIR_IS\RIR_IS_PrimarySource0_Receiver0_1_1.wav")
#geo_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\rir_shoebox.npz")
hybrid_data = np.load(r"C:\Masters\Hybrid\hybridsim\results\hybrid_rir.npz")
hybrid_rir = hybrid_data['arr_0'].astype(np.float64) 
t_geo = np.linspace(0, 2.0, 44100)


if rir.dtype != np.float32 and rir.dtype != np.float64:
    rir = rir.astype(np.float64) / np.iinfo(rir.dtype).max

t = np.arange(len(rir)) / fs
plt.plot(t, rir, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Raven RIR_RT')
plt.grid(True)
plt.show()

def octave_filter(signal, fs, fc, order=4):
    f_low = fc / np.sqrt(2)
    f_high = fc * np.sqrt(2)
    sos = butter(order, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, signal)

print("RAVEN sim")
for band in FREQ_BANDS: 
    band_rir = octave_filter(rir, fs = 44100, fc = band)
    sch_band = schroeder_integration(band_rir)
    RT_60_band = calculate_rt60(sch_band, fs = 44100)
    print(f"Freq (Hz): {band} RT60: {RT_60_band}")

print("hybrid")
for band in FREQ_BANDS: 
    band_rir = octave_filter(hybrid_rir, fs = 44100, fc = band)
    sch_band = schroeder_integration(band_rir)
    RT_60_band = calculate_rt60(sch_band, fs = 44100)
    print(f"Freq (Hz): {band} RT60: {RT_60_band}")

print(f"RAVEN RIR length:  {len(rir)/44100:.3f} s")
print(f"Hybrid RIR length: {len(hybrid_rir)/44100:.3f} s")