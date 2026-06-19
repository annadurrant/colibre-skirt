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
from speclite import filters
import astropy.units as u
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Convert SKIRT integrated r-band output into hdf5 dataset."
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
    "--IDs",
    type=int,
    nargs="+",
    default=-1,
)

parser.add_argument(
    "--nproc",
    type=int,
    default=64,
)

parser.add_argument(
    "--aperture",
    type=int,
    help="Aperture size [kpc] (default: 0, all bound particles).",
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

catalogue_file = params["InputFilepaths"]["catalogueFile"].format(simPath=simPath,snap_nr=args.snap)
output_filepath = params["OutputFilepaths"]["GalaxyLuminositiesFilepath"].format(simPath=simPath,snap_nr=args.snap)

with h5.File(catalogue_file) as fi:
    halo_IDs = fi["InputHalos/HaloCatalogueIndex"][()]
fi.close()

#####################################
########### Define filter ###########
#####################################
def apply_filter(
    wavelengths, # micron
    f_lambda, # flux per wavelength in W/m2/micron
    filter_name, 
):

    filter_throughput = np.loadtxt(filters_table + filter_name + ".dat")

    myfilter = filters.FilterResponse(
        wavelength=filter_wavelengths[:,0],
        response=filter_throughput[:,1],
        meta=dict(group_name="JWST",
                band_name=filter_name.split("_")[-1])
    )

    mag = myfilter.get_ab_magnitude(
        f_lambda * u.W / u.m**2 / u.micron,
        wavelengths * u.micron,
    )

    lum = 10 ** (-0.4 * mag)
    
    return lum


#############################################
###### Galaxy luminosity calculation ########
#############################################
def loop_luminosity(
    idx,
    filter_name="JWST_F444W",
    aperture_name="tot"
):
    idx = int(idx)
    dset_id = np.where(halo_IDs == idx)[0][0] 

    sed_file = np.loadtxt(SKIRToutputFilePath + f"/snap{args.snap}_ID{idx}_SED_{aperture_name}_sed.dat")
    wavelengths = sed_file[:,0] # in micron
    attenuated_luminosity = sed_file[:,1] # in W/m2/micron
    intrinsic_luminosity = sed_file[:,2] # in W/m2/micron

    # run filter over seds
    intrinsic_luminosity = apply_filter(wavelengths,intrinsic_sed,filter_name)
    attenuated_luminosity = apply_filter(wavelengths,attenuated_sed,filter_name)

    return (dset_id, intrinsic_luminosity, attenuated_luminosity)

print("Aperture size [kpc]:", aperture)

if aperture == 0:
    aperture_name = "tot"
    group_name = "BoundSubhalo"
else:
    aperture_name = f"{aperture}kpc"
    group_name = f"ProjectedAperture/{aperture}kpc/projz"


# Get intrinsic values from SOAP for faint dust-free objects
with h5.File(catalogue_file) as fi:
    soap_dset = fi[f"{group_name}/CorrectedStellarLuminosity"]
    attributes = {}
    for key in soap_dset.attrs:
        if key == "Description":
            attributes[key] = "Total stellar luminosity for a top hat UV band [1450-1550 A], computed with SKIRT."
        else:
            attributes[key] = soap_dset.attrs[key]
    
    intrinsic_luminosities = soap_dset[()][:,0]
fi.close()

with h5.File(output_filepath) as fi:
    # intrinsic_luminosities = fi["BoundSubhalo/IntrinsicUVLuminosity"][()]
    beta_slopes = fi["BoundSubhalo/BetaSlope_DustFree"][()]
fi.close()

# Create arrays to store results
attenuated_luminosities = np.copy(intrinsic_luminosities)
extinction = np.zeros_like(intrinsic_luminosities)

# Loop over SKIRT IDs
sample_IDs = np.loadtxt(sampleFilepath + f"sample_{args.snap}.txt")[:,0]

with multiprocessing.Pool(processes=args.nproc) as pool:
    results = pool.map(partial(loop_luminosity,aperture_name=aperture_name), sample_IDs)

for dset_id, lum_int, lum_att, beta in results:
    intrinsic_luminosities[dset_id] = lum_int
    attenuated_luminosities[dset_id] = lum_att
    extinction[dset_id] = -2.5 * np.log10( lum_att / lum_int )
    beta_slopes[dset_id] = beta

print("Finished collecting SKIRT data and extinction factors.", flush=True)

# Create hdf5 file and save data
output_fi = h5.File(output_filepath,"a")

grp = output_fi.require_group(group_name)

try:
    del grp["IntrinsicUVLuminositySKIRT"]
    del grp["AttenuatedUVLuminosity"]
    del grp["UVExtinction"]
    del grp["BetaSlope"]
except:
    print("")

dset = grp.create_dataset("IntrinsicUVLuminositySKIRT",data=intrinsic_luminosities)
for attribute in attributes:
    dset.attrs[attribute] = attributes[attribute]

dset = grp.create_dataset("AttenuatedUVLuminosity",data=attenuated_luminosities)
for attribute in attributes:
    dset.attrs[attribute] = attributes[attribute]

grp.create_dataset("UVExtinction",data=extinction)
grp.create_dataset("BetaSlope",data=beta_slopes)

output_fi.close()

print("Done.", flush=True)

