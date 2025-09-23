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
catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=args.snap)
output_filepath = params['OutputFilepaths']['GalaxyLuminositiesFilepath'].format(simPath=simPath,snap_nr=args.snap)

sample_IDs = np.loadtxt(sampleFilepath + f'/sample_{args.snap}.txt',usecols=[0],dtype=int)

with h5.File(catalogue_file,'r') as fi:
    halo_IDs = fi['InputHalos/HaloCatalogueIndex'][()]
fi.close()

def loopSml(id):
    # Compute mass weighted smoothing lengths from particle files
    
    dset_id = np.where(halo_IDs == id)[0][0]
    stars_file = np.loadtxt(storeParticlesPath + f'/snap{args.snap}_ID{id}_stars.txt').T

    minitials = stars_file[5]
    smls = stars_file[3]
    massweighted_sml = minitials * smls / np.sum(minitials)
    smooothingLength = np.mean(massweighted_sml)

    return (dset_id, smooothingLength)

def loopHMR(id):
    dset_id = np.where(halo_IDs == id)[0][0]
    gas_file = np.loadtxt(storeParticlesPath + f'/snap{args.snap}_ID{id}_stars.txt')

    if len(gas_file) > 1:
        gas_file = np.atleast_2d(gas_file)

        dust_r = np.sqrt(gas_file[:, 0]**2 + gas_file[:, 1]**2 + gas_file[:, 2]**2) * 1e-3 # In kpc
        dust_m = np.sum(gas_file[:, 10:16], axis = 1) # In Msun

        dustMasses_sorted = dust_m[np.argsort(dust_r)]

        idx_halfmass = np.min(np.argwhere((np.cumsum(dustMasses_sorted) / np.sum(dustMasses_sorted)) >= 0.5))

        dustHalfMassRadius = np.sort(dust_r)[idx_halfmass]

        dustHalfMass = (np.sum(dust_m) / 2.)

        SigmaDust = dustHalfMass / (np.pi * dustHalfMassRadius**2) # In solar masses / kpc^2

    else:
        SigmaDust = 0

    return (dset_id, SigmaDust)


if __name__ == "__main__":
    
    # Compute mass-weighted mean Sml  
    # SmoothingLengths = np.zeros_like(halo_IDs, dtype=float)

    # with multiprocessing.Pool(processes=64) as pool:
    #     results = pool.map(loopSml, sample_IDs)

    # for idx, value in results:
    #     SmoothingLengths[idx] = value

    # output_fi = h5.File(output_filepath,'a')
    # output_fi.create_dataset('BoundSubhalo/InitialMassWeightedSmoothingLength',data=SmoothingLengths)
    # output_fi.close()


    # Compute surface density of dust within HMR of dust
    Sigmas = np.zeros_like(halo_IDs, dtype=float)

    with multiprocessing.Pool(processes=64) as pool:
        results = pool.map(loopHMR, sample_IDs)

    for idx, value in results:
        Sigmas[idx] = value

    output_fi = h5.File(output_filepath,'a')

    output_fi.create_dataset('BoundSubhalo/DustSurfaceDensity',data=Sigmas)

    output_fi.close()

    # test
    # with h5.File(catalogue_file,'r') as fi:
    #     pc = 3.08567758149e+18
    #     Msun = unyt.Msun.to_value('g')

    #     hmrs = ( fi['ExclusiveSphere/50kpc/DustSmallGrainMass'][()] + fi['ExclusiveSphere/50kpc/DustLargeGrainMass'][()] ) / \
    #         (2 * np.pi * pow(fi[f'ExclusiveSphere/50kpc/HalfMassRadiusGas'][()], 2 ) )

    #     convert = fi[f'ExclusiveSphere/50kpc/StellarMass'].attrs['Conversion factor to physical CGS (including cosmological corrections)'][0]/Msun / \
    #         (fi[f'ExclusiveSphere/50kpc/HalfMassRadiusStars'].attrs['Conversion factor to physical CGS (including cosmological corrections)'][0]/pc/1e3)**2
        
    #     hmrs *= convert #Msun/kpc^2

    #     print(hmrs[hmrd>0])
    # fi.close()

    print('Done!')
