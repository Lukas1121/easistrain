from id15_NXstress import NXstressFromRaw
import os

# Determine the absolute directory path of test.py
dir_path = os.path.abspath(os.path.dirname(__file__))

file_path = os.path.join(dir_path, "../data/Ni_Ring_0001.h5")
det_calib_file_angle = os.path.join(dir_path, "../data/angleCalib.h5")
det_calib_file_energy = os.path.join(dir_path, "../data/energyCalib.h5")

nx_stress = NXstressFromRaw(
    file_path=file_path,
    det_calib_file_angle=det_calib_file_angle,
    det_calib_file_energy=det_calib_file_energy,
    with_cradle=False,
    lattice="lattice",
    phase_name="phase_name",
    scanNbForRotation=80,
    experimental_identifier="experimental_identifier",
    collection_identifier="collection_identifier",
    test_script=True,
)

nx_stress.main()
