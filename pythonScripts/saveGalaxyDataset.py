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
    help="<Required> Snapshot number.",
)

parser.add_argument(
    "outputDir",
    type=str,
    help="Name of output directory.",
)

parser.add_argument(
    "--SOAP",
    action="store_true",
    help="Also store data to SOAP catalogue.",
)

args = parser.parse_args()

# Define filepaths from parameter file
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = 'vimf_SKIRT_parameters.yml'
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)

simPath = params['InputFilepaths']['simPath'].format(simName=args.simName)
sampleFilepath = params['OutputFilepaths']['sampleFolder'].format(simPath=simPath)
SKIRToutputFilePath = params['OutputFilepaths']['SKIRToutputFilePath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation'])
SKIRToutputFilePath += args.outputDir

# Load SOAP catalogue and required attributes for a SOAP dset
catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=args.snap)

with h5.File(catalogue_file) as fi:
    halo_IDs = fi['InputHalos/HaloCatalogueIndex'][()]
    
    soap_dset = fi['BoundSubhalo/CorrectedStellarLuminosity']
    attributes = {}
    for key in soap_dset.attrs:
        if key == 'Description':
            attributes[key] = 'Total dust-attenuated stellar luminosity in the GALEX FUV band, computed with SKIRT.'
        else:
            attributes[key] = soap_dset.attrs[key]
    
    halo_luminosities = np.array([
        soap_dset[()][:,4],
        fi['ProjectedAperture/50kpc/projz/CorrectedStellarLuminosity'][()][:,4]
    ]).T
    
fi.close()

halo_luminosities_with_skirt = np.copy(halo_luminosities)

# Read in SKIRT data and add to halo data
# skirt_files = os.listdir(SKIRToutputFilePath)
sample_file = np.loadtxt(sampleFilepath + f'sample_{args.snap}.txt')

extinction = np.zeros_like(halo_luminosities_with_skirt)

for idx in sample_file[:,0]:
    idx=int(idx)

    file = SKIRToutputFilePath + f'/snap{args.snap}_ID{idx}_SED_tot_sed.dat'
    file_50 = SKIRToutputFilePath + f'/snap{args.snap}_ID{idx}_SED_50kpc_sed.dat'
    files = [file, file_50]
    
    if os.path.isfile(file):
        dset_id = np.where(halo_IDs == idx)[0][0]

        for i,fi in enumerate(files):
            skirt_lum =  1e12 * np.loadtxt(fi,usecols=1) / 3631 #Jy
            halo_luminosities_with_skirt[dset_id][i] = skirt_lum
            extinction[dset_id][i] = -2.5 * np.log10( skirt_lum / halo_luminosities[dset_id][i] )

print('Finished collecting SKIRT data and extinction factors.', flush=True)

# Create hdf5 file and save data
output_filepath = params['OutputFilepaths']['GalaxyLuminositiesFilepath'].format(simPath=simPath,snap_nr=args.snap)
output_fi = h5.File(output_filepath,'a')

grp_names = ['BoundSubhalo','ProjectedAperture/50kpc/projz']

for gi, grp_name in enumerate(grp_names):

    grp = output_fi.require_group(grp_name)

    try:
        dset = grp['FUVStellarLuminosity']
        dset[...] = halo_luminosities_with_skirt[:,gi]

    except:
        dset = grp.create_dataset('FUVStellarLuminosity',data=halo_luminosities_with_skirt[:,gi])

    for attribute in attributes:
        dset.attrs[attribute] = attributes[attribute]

    try:
        dset_extinct = grp['Extinction']
        dset_extinct[...] = extinction[:,gi]
    except:
        dset_extinct = grp.create_dataset('FUVExtinction',data=extinction[:,gi])

output_fi.close()

if args.SOAP == True:
    with h5.File(catalogue_file,'a') as dst_fi, h5.File(output_filepath,'a') as src_fi:
        src_fi.copy(src_fi['ProjectedAperture/50kpc/projz/FUVStellarLuminosity'],dst_fi['ProjectedAperture/50kpc/projz'],'CorrectedStellarLuminosityWithSKIRT')
    src_fi.close()
    dst_fi.close()

print('Done.', flush=True)