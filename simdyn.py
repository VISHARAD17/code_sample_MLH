import numpy as np
import module_shared as sh
from class_simulation import Simulation
from module_input import read_input_file
from module_radiation_scattering import calculate_radiation_scattering
from module_irf import calculate_IRF_from_RadiationFile, plot_IRF, verify_IRF_calculation
from module_solve import initialize_simulation, simulate
from module_validation import FK_HST_validate, validate_scattering_forces, radiation_force_validation, RAO_validation
from module_validation import radiation_force_validation
from module_output import generate_output
from websim import plot_ssvar,LoadFile
import time
import matplotlib.animation as Animate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import sys

def main_sim(time_stamp,filenames):

    #try:

        start_time = time.time()
        sh.time_stamp = time_stamp
        sh.filenames = filenames

        inp_fname = sh.filenames[3]
        inp_fname = "KCS.inp" # Uncomment for independent run

        sh.ulog = open('Simdyn.ulog','w')

        # Initialize simulation instance
        sh.sim = Simulation()

        #Initialize spectrum & spread parameter list
        sh.spectrum = []

        # Read input file and update sh.sim
        # Code in module_input.py
        read_input_file(inp_fname)

        # Create grid and compute hydrostatics
        # Code in class_grid.py, class_panels.py and class_hydrostatics.py
        for i in range(sh.sim.nVessels):
            sh.sim.Vessels[i].Grid.LoadFile()
            sh.sim.Vessels[i].Hydrostatics.compute_hydrostatics(sh.sim.Vessels[i].Grid)

            # sh.sim.Vessels[i].Grid.gridtest()             #<---- print out class panel variables to debug file "debugpanels.txt"
            # sh.sim.Vessels[i].Grid.Write_UWPanel_GDF()    #<--- write the underwater panels into a GDF file
            # sh.sim.Vessels[i].Hydrostatics.hydrotest()    #<---- print out class Hydrostatics variables

        # sh.sim.Vessels[0].Grid.DispMesh(100)              #<---Display Underwater mesh wrt global coordintes, 100 is scale of mesh

        # Setup Wave Environment
        # Code in class_waves.py
        for i in range(0, len(sh.spectrum)):
            if sh.spectrum[i].Name == 'regular':
                sh.sim.Environment.Waves.add_regular_wave(sh.spectrum[i])
            else:
                sh.sim.Environment.Waves.add_irregular_wave(sh.spectrum[i])

        # Verify wave environment by plotting wave elevation over space at t = 1 sec
        # x = np.linspace(0, 500, 250)
        # y = np.linspace(0, 500, 250)
        # x, y, elevation = sh.sim.Environment.Waves.calculate_wave_elevation(x, y, 0)
        # sh.sim.Environment.Waves.plot_waves(x,y,elevation)

        # Wave Animation --> Uncomment for wave animation
        # x = np.linspace(0, 500, 250)
        # y = np.linspace(0, 500, 250)
        # x, y, elevation = sh.sim.Environment.Waves.calculate_wave_elevation(x, y, 0)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_zlim(-3,3)
        # plot = [ax.plot_surface(x, y, elevation, cmap=cm.coolwarm, linewidth=0, antialiased=False)]
        #anim = Animate.FuncAnimation(fig,sh.sim.Environment.Waves.wave_animation,frames=5,fargs=(plot,ax),interval=5,blit=False)

        # anim.save("Irregular_PM.gif",writer="imagemagick",fps=10)

        # Read added mass A(omega) and radiation damping B(omega)
        # and compute impulse response function K(t)
        # Code in module_irf.py

        calculate_IRF_from_RadiationFile()

        # plot_IRF(4,4)                     #<--- Plot a particular IRF
        # verify_IRF_calculation(4,4)       #<--- Plot recalculated added mass and radiation damping

        # Initialize simulation
        # Code in module_solve.py
        initialize_simulation()

        # Simulate the solution
        # Code in module_solve.py that iteratively calls module_force.py
        simulate()
        # FK_HST_validate()  # <-- uncomment to run FK_validation comment simulate()

        # Scattering force verification --> Uncomment to verify (Set ULEN = 1)
        # validate_scattering_forces()

        # Write results to output files
        # Code in module_output.py

        # Close log file and finish execution
        plot_ssvar()
        print("Loading Visualization files...")
        sh.log_data = "Loading Visual files..."
        LoadFile(sh.filenames[0][0],True)        #for websim
        print("Visualization files loaded.")
        sh.log_data = "Simulation complete!! Hit refresh to load results."
        sh.ulog.close()
        end_time = time.time()
        print("execution time: ", end_time - start_time)
        print("Done")

    #except:

     #   sh.plot_filenames = False
      #  return False

    #else:

        return True

if __name__ == '__main__':

    # Set command line flag in module_shared to 1. This flag
    # causes the program to read other input file names from the
    # main input file specified as a command line argument. In a
    # default run, these are read from flask application and the
    # corresponding lines in main input file are ignored.
    sh.cmd_line_flag = 1

    start_time = time.time()

    # Read input file name from command line arguments
    narg = len(sys.argv)
    if narg != 2:
        print('ERROR: Program expects a single argument of input file name (no spaces) on the command line')
        exit()
    inp_fname = str(sys.argv[1])

    # inp_fname = 'KCS.inp'
    sh.ulog = open('Simdyn.ulog', 'w')

    # Initialize simulation instance
    sh.sim = Simulation()

    # Initialize spectrum & spread parameter list
    sh.spectrum = []

    # Read input file and update sh.sim
    # Code in module_input.py
    read_input_file(inp_fname)

    # Create grid and compute hydrostatics
    # Code in class_grid.py, class_panels.py and class_hydrostatics.py
    for i in range(sh.sim.nVessels):
        sh.sim.Vessels[i].Grid.LoadFile()
        sh.sim.Vessels[i].Hydrostatics.compute_hydrostatics(sh.sim.Vessels[i].Grid)

        # sh.sim.Vessels[i].Grid.gridtest()             #<---- print out class panel variables to debug file "debugpanels.txt"
        # sh.sim.Vessels[i].Grid.Write_UWPanel_GDF()    #<--- write the underwater panels into a GDF file
        # sh.sim.Vessels[i].Hydrostatics.hydrotest()    #<---- print out class Hydrostatics variables

    # sh.sim.Vessels[0].Grid.DispMesh(100)              #<---Display Underwater mesh wrt global coordintes, 100 is scale of mesh

    # Calculate Radiation and Diffraction
    calculate_radiation_scattering()

    # Setup Wave Environment
    # Code in class_waves.py
    for i in range(0, len(sh.spectrum)):
        if sh.spectrum[i].Name == 'regular':
            sh.sim.Environment.Waves.add_regular_wave(sh.spectrum[i])
        else:
            sh.sim.Environment.Waves.add_irregular_wave(sh.spectrum[i])

    # Verify wave environment by plotting wave elevation over space at t = 1 sec
    # x = np.linspace(0, 500, 250)
    # y = np.linspace(0, 500, 250)
    # x, y, elevation = sh.sim.Environment.Waves.calculate_wave_elevation(x, y, 0)
    # sh.sim.Environment.Waves.plot_waves(x,y,elevation)

    # Wave Animation --> Uncomment for wave animation
    # x = np.linspace(0, 500, 250)
    # y = np.linspace(0, 500, 250)
    # x, y, elevation = sh.sim.Environment.Waves.calculate_wave_elevation(x, y, 0)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_zlim(-3,3)
    # plot = [ax.plot_surface(x, y, elevation, cmap=cm.coolwarm, linewidth=0, antialiased=False)]
    # anim = Animate.FuncAnimation(fig,sh.sim.Environment.Waves.wave_animation,frames=5,fargs=(plot,ax),interval=5,blit=False)

    # anim.save("Irregular_PM.gif",writer="imagemagick",fps=10)

    # Read added mass A(omega) and radiation damping B(omega)
    # and compute impulse response function K(t)
    # Code in module_irf.py
    calculate_IRF_from_RadiationFile()

    # plot_IRF(4,4)                     #<--- Plot a particular IRF
    # verify_IRF_calculation(4,4)       #<--- Plot recalculated added mass and radiation damping

    # Initialize simulation
    # Code in module_solve.py
    initialize_simulation()

    # radiation force validation
    # code in module_force

    # FK_HST_validate()                     # <-- uncomment to run FK_validation
    # validate_scattering_forces()          # <-- Uncomment to verify scattering forces (Set ULEN = 1)
    # radiation_force_validation(1)         # <-- Validation of radiation forces due to velocity in a particular mode

    # Simulate the solution
    # Code in module_solve.py that iteratively calls module_force.py
    simulate()

    # RAO_validation(5)                       # <-- uncomment to run RAO_validation

    # Write results to output files
    # Code in module_output.py
    generate_output()

    # Close log file and finish execution
    sh.ulog.close()

    end_time = time.time()
    print("execution time: ", end_time - start_time)