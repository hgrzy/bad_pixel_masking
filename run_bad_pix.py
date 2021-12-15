import numpy as np 
from astropy.io import fits
import os
import fnmatch

from bad_pix import bad_pix

#making dictionary of bad pixel objects for every run in a day

bad_pix_dict = dict() #dictionary of bad pixel objects

os.startfile("O:")
directory = r'O:\HCD_Data\Speedster_data\Speedster550_data\1720762\21_10_19'
for filename in os.listdir(directory):
    if fnmatch.fnmatch(filename, 'Fe55_*'):
        directory2 = directory + "\\" + filename
        for filename2 in os.listdir(directory2):
            if fnmatch.fnmatch(filename2, 'output'):
                directory3 = directory2 + "\\" + filename2
                for fitsfile in os.listdir(directory3):
                    if fnmatch.fnmatch(fitsfile, 'Smooth_Dark*'):
                        fits_loc = directory3 + '\\' + fitsfile
                        bad_pix_dict[directory2[-5:]] = bad_pix(fits_loc)



