"""
Pipeline script to run both wave and geometrical simulations
and combine them into a hybrid model 
"""

import os
import sys 
import numpy as np

HYBRID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

sys.path.append(os.path.join(HYBRID_DIR, "DGsim"))
sys.path.append(os.path.join(HYBRID_DIR, "RayroomProject"))

from hybridsim.scenario2.geo_wrapper  import run_geometric
from hybridsim.scenario2.wave_wrapper import run_wave
from calibration import create_hybrid

def main():

    source_pos   = [3.04, 2.59, 1.62]
    rec_pos      = np.array([4.26, 1.76, 1.62])
    rir_duration = 0.1
    max_freq     = 50

    mesh_input = os.path.join(
        ROOT, "DGsim", "examples", "scenario2", 
        "oculus_scenario_2_lc05"
    )

    # ── Verify mesh exists ──
    if not os.path.exists(f"{mesh_input}.msh"):
        raise FileNotFoundError(f"Mesh not found: {mesh_input}.msh")

    # ── Run geometric simulation ──
    print("\n" + "="*50)
    print("STEP 1: GEOMETRIC SIMULATION")
    print("="*50)
    geo_res = run_geometric(
        rec_pos      = rec_pos,
        source_pos   = source_pos,
        n_rays       = 200,
        max_hops     = 150,
        rir_duration = rir_duration,
        ism_order    = 2
    )

    # ── Run wave simulation ──
    print("\n" + "="*50)
    print("STEP 2: WAVE SIMULATION")
    print("="*50)
    wave_res = run_wave(
        mesh_input   = mesh_input,
        max_freq     = max_freq,
        recx         = rec_pos[0],
        recy         = rec_pos[1],
        recz         = rec_pos[2],
        source_pos   = source_pos,
        rir_duration = rir_duration
    )

    # ── Create hybrid ──
    print("\n" + "="*50)
    print("STEP 3: HYBRID COMBINATION")
    print("="*50)
    hybrid = create_hybrid(
        geo_res      = geo_res,
        wave_res     = wave_res,
        crossover_hz = 100,
        fs           = 44100
    )

    hybrid_res_left = hybrid["hybrid_rir_left"]
    hybrid_res_right = hybrid["hybrid_rir_right"]
    hybrid_res_mono = hybrid["hybrid_rir_mono"]
    wave_calibrated_mono = hybrid["wave_calibrated_mono"]
    geo_scaled_mono = hybrid["rir_g_scaled_mono"]
    eta_mono = hybrid["eta_mono"]
    eta_left = hybrid["eta_left"]
    eta_right = hybrid["eta_right"]
    eta_avg = hybrid["eta_av"]

    # ── Save results ──
    output_dir = os.path.join(
        ROOT, "hybridsim", "scenario1", "results"
    )
    os.makedirs(output_dir, exist_ok=True)

    np.savez(
        os.path.join(output_dir, "hybrid_pipeline_test.npz"),
        hybrid_left  = hybrid_res_left,
        hybrid_right = hybrid_res_right,
        hybrid_mono = hybrid_res_mono,
        rir_total    = geo_res["rir_total"],
        wave_calibrated = wave_calibrated_mono, 
        geo_scaled = geo_scaled_mono, 
        eta_mono = eta_mono,
        eta_left = eta_left, 
        eta_right = eta_right, 
        eta_avg = eta_avg,
        wave_ir      = wave_res["IR_resampled"],
        td           = geo_res["td"],
        fs           = 44100
    )

    return wave_res, geo_res, hybrid_res_left, hybrid_res_right

if __name__ == "__main__":
    wave_res, geo_res, hybrid_l, hybrid_r = main()
