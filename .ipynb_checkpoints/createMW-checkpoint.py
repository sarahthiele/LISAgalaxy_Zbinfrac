#=========================================================================
# This script creates LISA DWD galaxies across 15 metallicity bins,
# incorporating the metallicity-dependent binary fraction as
# discussed in Thiele et al. (2021).
#
# Authors: Sarah Thiele & Katelyn Breivik
# Last updated: Oct 14th, 2021
#=========================================================================

import numpy as np
from astropy import constants as const
from astropy import units as u
import astropy.coordinates as coords
from astropy.time import Time
import argparse
import postproc as pp


# Set constants:
# LEGWORK uses astropy units so we do also for consistency
G = const.G.value  # gravitational constant
c = const.c.value  # speed of light in m s^-1
M_sol = const.M_sun.value  # sun's mass in kg
R_sol = const.R_sun.value  # sun's radius in metres
sec_Myr = u.Myr.to('s')  # seconds in a million years
m_kpc = u.kpc.to('m')  # metres in a kiloparsec
L_sol = const.L_sun.value  # solar luminosity in Watts
Z_sun = 0.02  # solar metallicity
sun = coords.get_sun(Time("2021-04-23T00:00:00", scale='utc'))  # sun coordinates
sun_g = sun.transform_to(coords.Galactocentric)
sun_yGx = sun_g.galcen_distance.to('kpc').value
sun_zGx = sun_g.z.to('kpc').value
M_astro = 7070  # FIRE star particle mass in solar masses

parser = argparse.ArgumentParser()
parser.add_argument('--DWD-list', nargs='+', default=['He_He', 'CO_He', 'CO_CO', 'ONe_X'])
parser.add_argument('--path', default='./', help='path to COSMIC dat files')
parser.add_argument('--FIRE-path', default='./', help='path to FIRE.h5 data')
parser.add_argument('--lband-path', default='./', help='path to save LISA band DWD data')
parser.add_argument('--nproc', default=1, type=int, help='number of processes to allow if using on compute cluster')
parser.add_argument('--interfile', default='False', type=str, help='if True, saves DWD formation, mergers, and RLOF data')

args = parser.parse_args()

pp.save_full_galaxy(args.DWD_list, args.path, args.FIRE_path, args.lband_path, args.interfile, args.nproc)
