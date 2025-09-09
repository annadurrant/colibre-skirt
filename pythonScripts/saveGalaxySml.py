import numpy as np
from swiftsimio import load as load_snapshot
import unyt
import yaml
import argparse
import os
import h5py as h5
from tqdm import tqdm
import matplotlib.pyplot as plt

# Set simName
parser = argparse.ArgumentParser(
)

parser.add_argument(
    "simName",
    type=str,
    help="Simulation name.",
)

parser.add_argument(
    "snap",
    type=int,
    help="<Required> Snapshot number.",
)

args = parser.parse_args()

# Define filepaths from parameter file
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = 'vimf_SKIRT_parameters.yml'
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)

simPath = params['InputFilepaths']['simPath'].format(simName=args.simName)
sampleFilepath = params['OutputFilepaths']['sampleFolder'].format(simPath=simPath)
storeParticlesPath = params['OutputFilepaths']['storeParticlesPath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation']) # Folder where the .txt particle files are stored

sample_IDs = np.loadtxt(sampleFilepath + f'/sample_{args.snap}.txt',usecols=[0],dtype=int)

# Load SOAP catalogue and required attributes for a SOAP dset
catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=args.snap)

with h5.File(catalogue_file,'r') as fi:
    halo_IDs = fi['InputHalos/HaloCatalogueIndex'][()]
fi.close()

smooothingLengths = np.zeros_like(halo_IDs, dtype=float)

for id in sample_IDs:
    
    dset_id = np.where(halo_IDs == id)[0][0]
    stars_file = np.loadtxt(storeParticlesPath + f'/snap{args.snap}_ID{id}_stars.txt').T

    minitials = stars_file[5]
    smls = stars_file[3]
    massweighted_sml = minitials * smls / np.sum(minitials)
    smooothingLengths[dset_id] = np.mean(massweighted_sml)

# save to SKIRT hdf5 file
output_filepath = params['OutputFilepaths']['GalaxyLuminositiesFilepath'].format(simPath=simPath,snap_nr=args.snap)
output_fi = h5.File(output_filepath,'a')

output_fi.create_dataset('BoundSubhalo/InitialMassWeightedSmoothingLength',data=smooothingLengths)
output_fi.close()