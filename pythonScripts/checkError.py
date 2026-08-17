import numpy as np
from swiftsimio import load as load_snapshot
import unyt
import yaml
import argparse
import os
import h5py as h5
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing
from functools import partial
from scipy.interpolate import interp1d
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Set simName
parser = argparse.ArgumentParser(
    description="Convert SKIRT integrated FUV-luminosity output into hdf5 dataset."
)

parser.add_argument(
    "simName",
    type=str,
    help="Simulation name.",
)

parser.add_argument(
    "snap",
    type=int,
    help="Snapshot number.",
)

parser.add_argument(
    "outputDir",
    type=str,
    help="Name of output directory.",
)

parser.add_argument(
    "--IDs",
    type=int,
    nargs="+",
    default=-1,
)

args = parser.parse_args()

#############################################
#### Define filepaths from parameter file ###
#############################################
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = "vimf_SKIRT_parameters.yml"
with open(f"{dir_path}/{param_file}","r") as stream:
    params = yaml.safe_load(stream)

simPath = params["InputFilepaths"]["simPath"].format(simName=args.simName)
sampleFilepath = params["OutputFilepaths"]["sampleFolder"].format(simPath=simPath)
SKIRToutputFilePath = params["OutputFilepaths"]["SKIRToutputFilePath"].format(simPath=simPath,rotation=params["ModelParameters"]["rotation"])
SKIRToutputFilePath += args.outputDir


n_err = 0
redo_ids = []


if args.IDs != -1:
    IDs = args.IDs
else:
    IDs = np.loadtxt(sampleFilepath + f"sample_{args.snap}.txt")[:,0]

errors = np.zeros_like(IDs, dtype = float)

for i,idx in enumerate(IDs):

    stats_file = np.loadtxt(SKIRToutputFilePath + f"/snap{args.snap}_ID{int(idx)}_SED_tot_sedstats.dat")
    N = stats_file[0][1]
    w1 = stats_file[0][2]
    w2 = stats_file[0][3]
    err = np.sqrt( w2/w1**2 - 1/N )
    if err > 0.1:
        n_err += 1
        redo_ids.append(idx)
    errors[i] = err

print(f"snapshot {args.snap}")
print(f"There are {n_err} galaxies with R > 0.1.")
if n_err > 0:
    print("These are:")
    for idx in redo_ids:
        print(int(idx))

ids_with_errors = np.array([ IDs, errors]).T

header = "Column 1: Halo ID\n" + \
            "Column 2: Relative error R \n"

np.savetxt(sampleFilepath + f"/relative_error_{args.snap}.txt", ids_with_errors, fmt = ["%d", "%.6f"], header = header)