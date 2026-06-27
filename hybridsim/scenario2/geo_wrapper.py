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
    mat_carpet = get_material("custom_carpet")
    mat_panel  = get_material("custom_panel")

    h      = 3.3
    offset = 0.001

    # Floor (z=0)
    P1f = np.array([0.0,     0.0,     0.0])
    P2f = np.array([5.51566, 0.0,     0.0])
    P3f = np.array([6.21333, 4.01907, 0.0])
    P4f = np.array([0.0,     5.09763, 0.0])

    # Ceiling (z=3.3)
    P1c = np.array([0.0,     0.0,     h])
    P2c = np.array([5.51566, 0.0,     h])
    P3c = np.array([6.21333, 4.01907, h])
    P4c = np.array([0.0,     5.09763, h])


    right_dir    = P3f - P2f
    right_dir    = right_dir / np.linalg.norm(right_dir)
    right_normal = np.array([-right_dir[1], right_dir[0], 0.0])

    back_dir    = P4f - P3f
    back_dir    = back_dir / np.linalg.norm(back_dir)
    back_normal = np.array([-back_dir[1], back_dir[0], 0.0])

    walls = []


    # Floor — Scenario 2 has smaller carpet, so floor is hard surface
    walls.append(Wall("Floor",      [P1f, P2f, P3f, P4f], mat_wall))

    # Ceiling
    walls.append(Wall("Ceiling",    [P4c, P3c, P2c, P1c], mat_wall))

    # Front wall
    walls.append(Wall("Wall_Front", [P1c, P2c, P2f, P1f], mat_wall))

    # Right wall
    walls.append(Wall("Wall_Right", [P2c, P3c, P3f, P2f], mat_wall))

    # Back wall
    walls.append(Wall("Wall_Back",  [P3c, P4c, P4f, P3f], mat_wall))

    # Left wall
    walls.append(Wall("Wall_Left",  [P4c, P1c, P1f, P4f], mat_wall))


    walls.append(Wall("Panel_3", [
        np.array([5.82908, 4.08577, 2.7]) + offset * back_normal,
        np.array([4.64676, 4.291,   2.7]) + offset * back_normal,
        np.array([4.64676, 4.291,   0.0]) + offset * back_normal,
        np.array([5.82908, 4.08577, 0.0]) + offset * back_normal,
    ], mat_panel))

    walls.append(Wall("Panel_4", [
        np.array([2.73904, 4.62216, 2.7]) + offset * back_normal,
        np.array([1.55672, 4.8274,  2.7]) + offset * back_normal,
        np.array([1.55672, 4.8274,  0.0]) + offset * back_normal,
        np.array([2.73904, 4.62216, 0.0]) + offset * back_normal,
    ], mat_panel))

    walls.append(Wall("Panel_5", [
        np.array([offset, 1.77, 2.7]),
        np.array([offset, 0.57, 2.7]),
        np.array([offset, 0.57, 0.0]),
        np.array([offset, 1.77, 0.0]),
    ], mat_panel))

    walls.append(Wall("Carpet", [
        np.array([1.90566, 3.84, 0.001]),
        np.array([1.90566, 0.05, 0.001]),
        np.array([4.78566, 0.05, 0.001]),
        np.array([4.78566, 3.84, 0.001]),
    ], mat_carpet))

    room_center = np.array([3.0, 2.5, 1.65])
    print("\n=== WALL NORMAL CHECK ===")
    all_inward = True
    for wall in walls:
        v0 = np.array(wall.vertices[0])
        v1 = np.array(wall.vertices[1])
        v2 = np.array(wall.vertices[2])
        normal = np.cross(v1 - v0, v2 - v0)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-10:
            print(f"{wall.name:15s} ❌ DEGENERATE")
            continue
        normal = normal / norm_mag
        dot    = np.dot(normal, room_center - v0)
        status = "✅ INWARD" if dot > 0 else "❌ OUTWARD"
        if dot <= 0:
            all_inward = False
        print(f"{wall.name:15s} {status}")
    print(f"\nAll inward: {'✅' if all_inward else '❌ Fix outward walls!'}")

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
    rirs  = tracer.render(n_rays=n_rays,
            max_hops=max_hops,
            rir_duration=rir_duration,
            record_paths=False,
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


    hrtf_path = r"C:\Masters\HRTF\KEMAR_GRAS_EarSim_LargeEars_FreeFieldCompMinPhase_44kHz.sofa"
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