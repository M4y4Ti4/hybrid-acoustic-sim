"""
computes room acoustic parameters for the RIRs
"""
import numpy as np
from rayroom.analytics.acoustics import schroeder_integration, calculate_rt60, calculate_edt, _calculate_decay_time, calculate_drr
from rayroom.core.constants import FREQ_BANDS
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG

#wave_data = np.load(r"C:\Masters\Hybrid\DGsim\examples\shoebox\output\TR_corrected_lc1_200Hz_2slc__06.npz")
wave_data = np.load(r"C:\Masters\Hybrid\DGsim\examples\shoebox\output\TR_corrected_lc1_200Hz_2slc__25.npz")

geo_data = np.load(r"C:\Masters\Hybrid\RayroomProject\examples\initial_testing\rir_shoebox_cal_newscale.npz")

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

print(RT60_wave)
print(EDT_geo,EDT_wave)