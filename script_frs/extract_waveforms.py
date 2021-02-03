# # Extract stream and save as mseed for each time stamp
from obspy import read, UTCDateTime, Stream
import os
import numpy as np
from glob import glob
import sys

WF_DIR_ROOT_500Hz = "/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz"
WF_DIR_ROOT_HAWK = "/home/gilbert_lab/cami_frs/hawk_data/sac_data_mps/"

def get_stream(tmin, tmax):    
    
    detst = Stream()
    
    # Get borehole waveforms
    pattern = os.path.join(WF_DIR_ROOT_500Hz, tmin.strftime("%Y%m%d"), "*DP*")
    print("# files for BH: %d" % len(glob(pattern)))
    for f in glob(pattern):
        tmp = read(f, starttime=tmin, endtime=tmax)
        detst += tmp
    
    # Get Hawk waveforms
    for channel in ["DPN", "DPE", "DPZ"]:
        pattern_hawk = os.path.join(WF_DIR_ROOT_HAWK, "*", "*", "*%s.D.%s*" % (channel, tmin.strftime("%Y%m%d")))
        if glob(pattern_hawk):
            print("# files for Hawk %s: %d" % (channel, len(glob(pattern_hawk))))
            for f in glob(pattern_hawk):
                if "1000Hz" in f:
                    continue
                detst += read(f, starttime=tmin, endtime=tmax)
                
    
    # pre-process
    detst.detrend("demean")
    
    return detst

eventstr = sys.argv[1]
print("Input string: %s" % eventstr)
event_time = UTCDateTime(eventstr)
prepick = 5.0
postpick = 10.0

fname = os.path.join("/home/genevieve.savard/borehole/energy_detector/deep_events/waveforms", "event_%s.mseed" % (event_time.strftime("%Y%m%d%H%M%S")))
#if not os.path.exists(fname):

starttime = event_time - prepick
endtime = event_time + postpick
print("Start time = %s\nEnd time = %s" % (starttime, endtime))

st = get_stream(tmin=starttime, tmax=endtime)

print("Writing stream to: %s" % fname)
st.write(fname, format="MSEED")

print(st.__str__(extended=True))