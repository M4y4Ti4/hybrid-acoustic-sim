"""
computes room acoustic parameters for the RIRs
"""
import numpy as np
from rayroom.analytics.acoustics import schroeder_integration, calculate_rt60, calculate_edt, _calculate_decay_time, calculate_drr
from rayroom.core.constants import FREQ_BANDS
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG


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
from scipy.signal import butter, sosfiltfilt

fs, rir = wavfile.read(r"C:\ITASoftware\Raven\RavenOutput\myShoeboxRoom5x4x320260421T161145\ImpulseResponses\2026-04-21\16.11.45\RIR_Combined\RIR_Combined_PrimarySource0_Receiver0_1_1.wav")

if rir.dtype != np.float32 and rir.dtype != np.float64:
    rir = rir.astype(np.float64) / np.iinfo(rir.dtype).max

t = np.arange(len(rir)) / fs
plt.plot(t, rir, linewidth=0.5)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Raven RIR_RT')
plt.grid(True)
plt.show()
"""