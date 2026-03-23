import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__)))

import os
import glob
import numpy
import scipy.io
import edg_acoustics
import time
from edg_acoustics.clean_results import post_process_output

def run_wave(mesh_input, max_freq, recx, recy, recz, source_pos, rir_duration,):

    rho0 = 1.213  # density of air at 20 degrees Celsius in kg/m^3
    c0 = 343  # speed of sound in air at 20 degrees Celsius in m/s

    BC_labels = {
        "brick": 14,
        "carpet": 13,
        "ceiling": 11,
    }  # predefined labels for boundary conditions. please assign an arbitrary int number to each type of boundary condition, e.g. hard wall, carpet, panel. The number should be unique for each type of boundary condition and should match the physical surface number in the .geo mesh file. The string should be the same as the material name in the .mat file (at least for the first few letters).

    real_valued_impedance_boundary = [
        {"label":   14, "RI": 0.9592},
        {"label":  13, "RI": 0.9849},
        {"label": 11, "RI": 0.9592},
    ]# extra labels for real-valued impedance boundary condition, if needed. The label should be the similar to the label in BC_labels. Since it's frequency-independent, only "RI", the real-valued reflection coefficient, is required. If not needed, just clear the elements of this list and keep the empty list.

    mesh_used = mesh_input
    mesh_name = f"{mesh_used}.msh"  # name of the mesh file. The mesh file should be in the same folder as this script.
    monopole_xyz = numpy.array(source_pos)  # x,y,z coordinate of the source in the room
    freq_upper_limit = max_freq  # upper limit of the frequency content of the source signal in Hz. The source signal is a Gaussian pulse with a frequency content up to this limit.

    # Approximation degrees
    Nx = 4  # in space
    Nt = 4  # in time
    CFL = 0.5  # CFL number, default is 0.5.
    recx = numpy.array(recx)
    recy = numpy.array(recy)
    recz = numpy.array(recz)
    rec = numpy.vstack((recx, recy, recz))  # dim:[3,n_rec]

    impulse_length = rir_duration  # total simulation time in seconds
    save_every_Nstep = 1  # save the results every N steps
    temporary_save_Nstep = 500  # save the results every N steps temporarily during the simulation. The temporary results will be saved in the root directory of this repo.

    # load Boundary conditions and parameters
    BC_para = real_valued_impedance_boundary

    # mesh_data_folder is the current folder by default
    mesh_data_folder = os.path.split(os.path.abspath(__file__))[0]
    mesh_filename = os.path.join(mesh_data_folder, mesh_name)
    mesh = edg_acoustics.Mesh(mesh_filename, BC_labels)


    IC = edg_acoustics.Monopole_IC(monopole_xyz, freq_upper_limit)

    sim = edg_acoustics.AcousticsSimulation(rho0, c0, Nx, mesh, BC_labels)

    flux = edg_acoustics.UpwindFlux(rho0, c0, sim.n_xyz)
    AbBC = edg_acoustics.AbsorbBC(sim.BCnode, BC_para)

    sim.init_BC(AbBC)
    sim.init_IC(IC)
    sim.init_Flux(flux)
    sim.init_rec(
        rec, "scipy"
    )  # brute_force or scipy(default) approach to locate the receiver points in the mesh

    simulation_start = time.time() #tracking the time of the simulation

    tsi_time_integrator = edg_acoustics.TSI_TI(sim.RHS_operator, sim.dtscale, CFL, Nt=3)
    sim.init_TimeIntegrator(tsi_time_integrator)
    sim.time_integration(
        total_time=impulse_length,
        delta_step=save_every_Nstep,
        save_step=temporary_save_Nstep,
        format="mat",
    )

    simulation_elapsed = time.time() - simulation_start

    prec = sim.prec.squeeze()
    dt_sim = sim.time_integrator.dt
    fs_dg = 1/dt_sim
    halfwidth = float(sim.IC.halfwidth)
    source_xyz = sim.IC.source_xyz.squeeze()
    rec_xyz = rec.squeeze()


    processed_res = post_process_output(prec=prec, dt_sim=dt_sim, source_xyz=source_xyz, rec_xyz=rec_xyz, halfwidth=halfwidth, fs_target=44100)
    IR_resampled = processed_res["IR"]
    TF_resampled = processed_res["TF"]
    t_resampled = processed_res["t_resampled"]
    freqs = processed_res["freqs"]

    return {
        "prec": sim.prec,
        "dt": sim.time_integrator.dt,
        "runtime_seconds": simulation_elapsed,
        "source_xyz": sim.IC.source_xyz,
        "halfwidth": sim.IC.halfwidth,
        "rec": rec,
        "c0": c0,
        "rho0": rho0,
        "mesh_name": mesh_name,
        "Nx": Nx,
        "CFL": CFL,
        "N_tets": sim.N_tets,
        "impulse_length": impulse_length,
        "IR_resampled": IR_resampled,
        "TF_resampled": TF_resampled, 
        "t_resampled": t_resampled,
        "freqs": freqs

    }
