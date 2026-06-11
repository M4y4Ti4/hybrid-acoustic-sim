import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from rayroom.core.utils import sum_frequency_bands
from rayroom import Room, Source, Receiver, Window, Person, RayTracer, get_material, HybridRenderer
from rayroom.core.data_anal import plot_rir, plot_transfer_function, overlay_DG, plot_rir_per_band, plot_rir_components
from rayroom.room.visualize import plot_reverberation_time
import random
from rayroom.room import Wall
from rayroom.core.auralisation import load_hrtf, render_brir, plot_brir, get_hrir
from rayroom.core.constants import FREQ_BANDS

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_geometric(rec_pos, source_pos, n_rays, max_hops, rir_duration, ism_order):
  

    mat_wall   = get_material("wall")
    carpet = get_material("custom_carpet")
    mat_panel  = get_material("custom_panel")

    h = 3.3

    # Floor vertices (z=0)
    P1f = np.array([0.0,     0.0,     0.0])
    P2f = np.array([5.51566, 0.0,     0.0])
    P3f = np.array([6.21333, 4.01907, 0.0])
    P4f = np.array([0.0,     5.09763, 0.0])

    # Ceiling vertices (z=3.3)
    P1c = np.array([0.0,     0.0,     h])
    P2c = np.array([5.51566, 0.0,     h])
    P3c = np.array([6.21333, 4.01907, h])
    P4c = np.array([0.0,     5.09763, h])

    walls = []

    # Floor (normal pointing up +Z)
    walls.append(Wall("Floor",   [P1f, P2f, P3f, P4f], carpet))

    # Ceiling (normal pointing down -Z, reverse order)
    walls.append(Wall("Ceiling", [P4c, P3c, P2c, P1c], mat_wall))

    # Front wall  (y=0, between P1-P2)
    walls.append(Wall("Wall_Front", [P1c, P2c, P2f, P1f], mat_wall))

    # Right wall  (between P2-P3)
    walls.append(Wall("Wall_Right", [P2c, P3c, P3f, P2f], mat_wall))

    # Back wall   (between P3-P4)
    walls.append(Wall("Wall_Back",  [P3c, P4c, P4f, P3f], mat_wall))

    # Left wall   (x=0, between P4-P1)
    walls.append(Wall("Wall_Left",  [P4c, P1c, P1f, P4f], mat_wall))

    #creating acoustic panels as wall objects 

    offset = 0.001  # Push panels 1mm into room to avoid coplanar issues

    # ── Front wall panels (y=0 wall → panels at y=offset, normal +Y) ──
    walls.append(Wall("Panel_1", [
        np.array([0.5, offset, 2.165]),
        np.array([2.6, offset, 2.165]),
        np.array([2.6, offset, 0.0  ]),
        np.array([0.5, offset, 0.0  ]),
    ], mat_panel))

    walls.append(Wall("Panel_2", [
        np.array([3.41566, offset, 2.7]),
        np.array([4.01566, offset, 2.7]),
        np.array([4.01566, offset, 0.0]),
        np.array([3.41566, offset, 0.0]),
    ], mat_panel))

    walls.append(Wall("Panel_3", [
        np.array([4.79566, offset, 2.7]),
        np.array([5.39566, offset, 2.7]),
        np.array([5.39566, offset, 0.0]),
        np.array([4.79566, offset, 0.0]),
    ], mat_panel))

    # ── Panel 4 - Angled right wall ──
    # Right wall goes from P2(5.51566,0,0) to P3(6.21333,4.01907,0)
    # Wall direction vector (normalised)
    wall_dir = np.array([6.21333 - 5.51566, 4.01907 - 0.0, 0.0])
    wall_dir = wall_dir / np.linalg.norm(wall_dir)
    # Inward normal (rotate 90° CCW in XY plane)
    wall_normal = np.array([-wall_dir[1], wall_dir[0], 0.0])

    walls.append(Wall("Panel_4", [
        np.array([5.87996, 2.09862,  1.02]) + offset * wall_normal,
        np.array([5.66617, 0.867034, 2.5 ]) + offset * wall_normal,
        np.array([5.87996, 2.09862,  2.5 ]) + offset * wall_normal,
        np.array([5.66617, 0.867034, 1.02]) + offset * wall_normal,
    ], mat_panel))

    # ── Back wall panels (P3->P4 wall) ──
    # Back wall direction vector
    back_dir = np.array([0.0 - 6.21333, 5.09763 - 4.01907, 0.0])
    back_dir = back_dir / np.linalg.norm(back_dir)
    # Inward normal
    back_normal = np.array([-back_dir[1], back_dir[0], 0.0])

    walls.append(Wall("Panel_5", [
        np.array([5.82908, 4.08577, 2.7]) + offset * back_normal,
        np.array([4.64676, 4.291,   2.7]) + offset * back_normal,
        np.array([4.64676, 4.291,   0.0]) + offset * back_normal,
        np.array([5.82908, 4.08577, 0.0]) + offset * back_normal,
    ], mat_panel))

    walls.append(Wall("Panel_6", [
        np.array([2.73904, 4.62216, 2.7]) + offset * back_normal,
        np.array([1.55672, 4.8274,  2.7]) + offset * back_normal,
        np.array([1.55672, 4.8274,  0.0]) + offset * back_normal,
        np.array([2.73904, 4.62216, 0.0]) + offset * back_normal,
    ], mat_panel))

    walls.append(Wall("Panel_7", [
        np.array([1.35967, 4.86161, 2.7]) + offset * back_normal,
        np.array([0.17735, 5.06684, 2.7]) + offset * back_normal,
        np.array([0.17735, 5.06684, 0.0]) + offset * back_normal,
        np.array([1.35967, 4.86161, 0.0]) + offset * back_normal,
    ], mat_panel))

    # ── Left wall panels (x=0 wall → panels at x=offset, normal +X) ──
    walls.append(Wall("Panel_8", [
        np.array([offset, 4.52763, 2.7]),
        np.array([offset, 3.32763, 2.7]),
        np.array([offset, 3.32763, 0.0]),
        np.array([offset, 4.52763, 0.0]),
    ], mat_panel))

    walls.append(Wall("Panel_9", [
        np.array([offset, 1.77, 2.7]),
        np.array([offset, 0.57, 2.7]),
        np.array([offset, 0.57, 0.0]),
        np.array([offset, 1.77, 0.0]),
    ], mat_panel))

    room = Room(walls=walls, fs=44100)

    # 2. Add Objects

    # Source at (1, 1, 1.5)
    source    = Source("Speaker", source_pos, power=1.0)
    room.add_source(source)

    # Receiver (Microphone) at (4, 3, 1.5)
    receiver = Receiver("persona", rec_pos, radius=0.1)
    room.add_receiver(receiver)

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
            show_path_plot=False, 
            parallel = False)
    rir_array = rirs[receiver.name]

    rir_total, rir_bands = sum_frequency_bands(rir_array, fs = 44100) #band-pass and sum each frequency band to produce broadband RIR

    hist = tracer.last_histogram[receiver.name]
    hist_sorted = sorted(hist, key=lambda x: x[0])

    direct = hist_sorted[0]
    time_direct, amp_direct, is_ism_direct, az_direct, el_direct = direct
    amp_direct_scalar = np.mean(np.abs(np.array(amp_direct))) #taking the mean of the band amplitudes for broadband scaling 

    first_reflection = hist_sorted[1]
    time_first, amp_first, is_ism_first, az_first, el_direct = first_reflection
    t_geo = np.linspace(0, rir_duration, len(rir_total))


    MASTERS_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
        "rir_bands": rir_bands,
        "fs": fs, 
        "brir_l": brir_l, 
        "brir_r": brir_r,
        "gd": amp_direct,
        "td": time_direct,
        "tr": time_first,
        "t_geo": t_geo,
    }