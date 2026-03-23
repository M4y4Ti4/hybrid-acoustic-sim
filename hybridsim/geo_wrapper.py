import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import sum_frequency_bands
from rayroom import Room, Source, Receiver, Person, RayTracer, get_material, HybridRenderer
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band, plot_rir_components
from rayroom.room.visualize import plot_reverberation_time
import random
from rayroom.core.auralisation import load_hrtf, render_brir, plot_brir, get_hrir

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def run_geometric(rec_pos, source_pos, room_dim, n_rays, max_hops, rir_duration, ism_order):

    # 1. Create Room (Shoebox 5m x 4m x 3m)
    # Different materials for walls
    mats = {
        "floor": get_material("carpet"),
        "ceiling": get_material("drywall"),
        "front": get_material("drywall"),
        "back": get_material("drywall"),
        "left": get_material("drywall"),
        "right": get_material("drywall")
    }

    room = Room.create_shoebox(room_dim, materials=mats)
    # 2. Add Objects
    # Source at (1, 1, 1.5)
    source    = Source("Speaker", source_pos, power=1.0)
    room.add_source(source)

    # Receiver (Microphone) at (4, 3, 1.5)
    receiver = Receiver("persona", rec_pos, radius=0.09)
    room.add_receiver(receiver)

    # Plot Room BEFORE Simulation (Check geometry)
    print("Saving room visualization...")
    room.plot("room_layout.png", show=False)

    # 3. Run Simulation
    tracer = HybridRenderer(room)

    #setting source to a delta function: 
    fs = 44100
    impulse_length = 128   # 128 samples = ~2.9 ms
    delta_impulse = np.zeros(impulse_length)
    delta_impulse[0] = 1.0  # first sample is 1
    tracer.set_source_audio(source, delta_impulse)

    print("Starting simulation...")
    #tracer.generate_rir_only(source, n_rays=20000, max_hops=30)
    rirs, all_paths  = tracer.render(n_rays=n_rays,
            max_hops=max_hops,
            rir_duration=rir_duration,
            record_paths=True,
            interference=True,
            ism_order=ism_order,         # Enable Hybrid Mode
            show_path_plot=True, 
            parallel = False)
    
    rir_array = rirs[receiver.name]

    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR

    hist = tracer.last_histogram[receiver.name]
    
    MASTERS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    hrtf_path = os.path.join(MASTERS_DIR, "HRTF", "KEMAR_GRAS_EarSim_LargeEars_FreeFieldCompMinPhase_44kHz.sofa")
    hrtf = load_hrtf(hrtf_path, fs_target=44100)

    brir_l, brir_r, brir_bands_l, brir_bands_r = render_brir(
    histogram=hist,
    src_xyz=source_pos,
    rec_xyz=rec_pos,
    hrtf=hrtf,
    fs=fs,
    duration=rir_duration,
    interference=True)

    return {
        "rir_total": rir_total, 
        "fs": fs, 
        "brir_l": brir_l, 
        "brir_r": brir_r
    }