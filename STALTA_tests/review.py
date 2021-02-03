from obspy import read_inventory, read, read_events, Catalog
from pyrocko import obspy_compat
obspy_compat.plant()
import sys
from glob import glob
import os
import shutil

flist = glob(os.path.join("waveforms", "*.mseed"))
for file in flist:
    print(file)
    stream = read(file)
    return_tag, markers_out = stream.snuffle()

    if return_tag == "x":
        os.rename(file, file.replace("waveforms/", "bad/"))
        print("X: reject")
    elif return_tag == "q":
        os.rename(file, file.replace("waveforms/", "good/"))
        print("Q: keep")
