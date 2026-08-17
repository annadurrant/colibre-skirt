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
    help="Snapshot number.",
)

parser.add_argument(
    "outputDir",
    type=str,
    help="Name of output directory.",
)

parser.add_argument(
    "--nproc",
    type=int,
    default=64,
)

parser.add_argument(
    "--aperture",
    type=int,
    default=0,
    help="Aperture size [kpc] (default: 0, all bound particles).",
)

parser.add_argument(
    "--redshift",
    type=int,
    default=0,
    help="Redshift the SED accordingly (default: 0).",
)

parser.add_argument(
    "--filter_name",
    type=str,
    default="SDSS_r",
    help="Name of filter for SED integration (default: SDSS_r).",
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
correctedSnapshotFile = params["InputFilepaths"]["correctedSnapshotFile"].format(simPath=simPath, snap_nr=args.snap)

catalogue_file = params["InputFilepaths"]["catalogueFile"].format(simPath=simPath,snap_nr=args.snap)
output_filepath = params["OutputFilepaths"]["GalaxyLuminositiesFilepath"].format(simPath=simPath,snap_nr=args.snap)

filterTablePath = params["InputFilepaths"]["filterTablePath"]

with h5.File(catalogue_file) as fi:
    halo_IDs = fi["InputHalos/HaloCatalogueIndex"][()]
fi.close()

#####################################
########### Define filter ###########
#####################################
filter_name = args.filter_name

if "SDSS" in filter_name:
    band = filter_name.split("_")[-1]
    myfilter = filters.load_filter("sdss2010-" + band)

else:

    filter_throughput = np.loadtxt(filterTablePath + filter_name + ".dat")
    
    myfilter = filters.FilterResponse(
        wavelength= filter_throughput[:,0] * u.um / (1 + args.redshift) ,
        response=filter_throughput[:,1],
        meta=dict(group_name=filter_name.split("_")[0],
                band_name=filter_name.split("_")[-1])
    )


def apply_filter(
    wavelengths, # micron
    f_lambda, # flux per wavelength in W/m2/micron,
    myfilter,
):

    mag = myfilter.get_ab_magnitude(
        f_lambda * u.W / u.m**2 / u.um,
        wavelengths * u.um,
    )

    lum = 10 ** (-0.4 * mag)

    return lum


#############################################
###### Galaxy luminosity calculation ########
#############################################
def loop_luminosity(
    idx,
    filter_name="SDSS_r",
    aperture_name="tot"
):
    idx = int(idx)
    dset_id = np.where(halo_IDs == idx)[0][0] 

    sed_file = np.loadtxt(SKIRToutputFilePath + f"/snap{args.snap}_ID{idx}_SED_{aperture_name}_sed.dat")
    if sed_file.ndim > 1:
        wavelengths = sed_file[:,0] # in micron
        attenuated_sed = sed_file[:,1] # in W/m2/micron
        intrinsic_sed = sed_file[:,2] # in W/m2/micron

        # run filter over seds
        intrinsic_luminosity = apply_filter(wavelengths, intrinsic_sed, myfilter)
        attenuated_luminosity = apply_filter(wavelengths, attenuated_sed, myfilter)
    else:
        print(f"SED {SKIRToutputFilePath}/snap{args.snap}_ID{idx}_SED_{aperture_name}_sed.dat is 1D.")
        return (dset_id, 0, 0)

    return (dset_id, intrinsic_luminosity, attenuated_luminosity)
    
aperture = args.aperture

redshift = args.redshift

print("Aperture size [kpc]:", aperture)
print(f"Filter name: {filter_name}; redshift: {redshift}")

if aperture == 0:
    aperture_name = "tot"
    group_name = "BoundSubhalo"
else:
    aperture_name = f"{aperture}kpc"
    group_name = f"ProjectedAperture/{aperture}kpc/projz"


# Set luminosity band column
with h5.File(correctedSnapshotFile,"r") as fi:
    column_names = fi["SubgridScheme/NamedColumns/CorrectedLuminosities"][()]
fi.close()

column_names = np.array([c.decode("utf-8") for c in column_names])
try:
    SEL = column_names == filter_name
    band_index = np.argwhere(SEL == True)[0][0]
except:
    band_index = -1
    print(f"Cannot find filter name {filter_name} in SOAP catalogue.")


# Get intrinsic values from SOAP for faint dust-free objects
with h5.File(catalogue_file) as fi:
    soap_dset = fi[f"{group_name}/CorrectedStellarLuminosity"]
    attributes = {}
    for key in soap_dset.attrs:
        if key == "Description":
            attributes[key] = f"Total stellar luminosity for {filter_name} (redshifted: {redshift}), computed with SKIRT."
        else:
            attributes[key] = soap_dset.attrs[key]
    if band_index != -1:
        intrinsic_luminosities = soap_dset[()][:,band_index]
    else:
        intrinsic_luminosities = np.zeros_like(soap_dset[()][:,0])
fi.close()

# with h5.File(output_filepath, "r") as fi:
#     intrinsic_luminosities = fi[f"{group_name}/Intrinsic{filter_name}Luminosity"][()]
# fi.close()

# Create arrays to store results
attenuated_luminosities = np.copy(intrinsic_luminosities)
extinction = np.zeros_like(intrinsic_luminosities)

# Loop over SKIRT IDs
sample_IDs = np.loadtxt(sampleFilepath + f"sample_{args.snap}.txt")[:,0]

with multiprocessing.Pool(processes=args.nproc) as pool:
    results = pool.map(partial(loop_luminosity,aperture_name=aperture_name), sample_IDs)

for dset_id, lum_int, lum_att in results:
    if lum_int == 0:
        continue
    intrinsic_luminosities[dset_id] = lum_int
    attenuated_luminosities[dset_id] = lum_att
    extinction[dset_id] = -2.5 * np.log10( lum_att / lum_int )

print("Finished collecting SKIRT data and extinction factors.", flush=True)

# Create hdf5 file and save data
output_fi = h5.File(output_filepath,"a")

grp = output_fi.require_group(group_name)

try:
    del grp[f"Intrinsic{filter_name}LuminositySKIRT"]
    del grp[f"Attenuated{filter_name}Luminosity"]
    del grp[f"{filter_name}Extinction"]
except:
    print("")

dset = grp.create_dataset(f"Intrinsic{filter_name}LuminositySKIRT",data=intrinsic_luminosities)
for attribute in attributes:
    dset.attrs[attribute] = attributes[attribute]

dset = grp.create_dataset(f"Attenuated{filter_name}Luminosity",data=attenuated_luminosities)
for attribute in attributes:
    dset.attrs[attribute] = attributes[attribute]

grp.create_dataset(f"{filter_name}Extinction",data=extinction)

output_fi.close()

print("Done.", flush=True)

