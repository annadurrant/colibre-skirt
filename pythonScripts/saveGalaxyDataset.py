import numpy as np
from swiftsimio import load as load_snapshot
import unyt
import yaml
import argparse
import os
import h5py as h5

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

args = parser.parse_args()

# Define filepaths from parameter file
dir_path =  os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
param_file = 'vimf_SKIRT_parameters.yml'
with open(f'{dir_path}/{param_file}','r') as stream:
    params = yaml.safe_load(stream)

simPath = params['InputFilepaths']['simPath'].format(simName=args.simName)
SKIRToutputFilePath = params['OutputFilepaths']['SKIRToutputFilePath'].format(simPath=simPath,rotation=params['ModelParameters']['rotation'])
SKIRToutputFilePath += args.outputDir

# Collect SKIRT output files
skirt_files = os.listdir(SKIRToutputFilePath)
skirt_luminosities = {}
for file in skirt_files:
    if 'SED_50kpc' in file:
        id = int(file.split('_')[1].replace('ID',''))
        skirt_luminosities[id] = np.loadtxt(SKIRToutputFilePath + '/' + file,usecols=1)

# Load SOAP catalogue and required attributes for a SOAP dset
catalogue_file = params['InputFilepaths']['catalogueFile'].format(simPath=simPath,snap_nr=args.snap)

with h5.File(catalogue_file) as fi:
    halo_IDs = fi['InputHalos/HaloCatalogueIndex'][()]
    dset = fi['ExclusiveSphere/50kpc/StellarLuminosity']
    attributes = {}
    for key in dset.attrs:
        if key == 'Description':
            attributes[key] = 'Total dust-attenuated stellar luminosity in the GALEX FUV band, computed with SKIRT.'
        else:
            attributes[key] = dset.attrs[key]
fi.close()

# Create hdf5 file and save data
output_filepath = params['OutputFilepaths']['GalaxyLuminositiesFilepath'].format(simPath=simPath,snap_nr=args.snap)
output_fi = h5.File(output_filepath,'a')

luminosity_array = np.zeros_like(halo_IDs,dtype=float)
for id in skirt_luminosities:
    dset_id = np.where(halo_IDs == id)[0][0]
    luminosity_array[dset_id] = 1e12 * skirt_luminosities[id] / 3631 #Jy


try:
    dset = output_fi['FUVStellarLuminosity']
    dset[...] = luminosity_array

except:
    dset = output_fi.create_dataset('FUVStellarLuminosity',data=luminosity_array)

    for attribute in attributes:
        dset.attrs[attribute] = attributes[attribute]


output_fi.close()