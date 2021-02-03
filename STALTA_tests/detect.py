from obspy import read, Stream
import os
import sys
from glob import glob
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import coincidence_trigger
import pandas as pd
import numpy as np



def network_detect(stream, sta, lta, on_thresh, off_thresh=None, min_chans=8, min_sep=3., max_trigger_length=3.0):
    if not off_thresh:
        off_thresh = 0.5*on_thresh
    print("STA = %f, LTA = %f, on_thresh = %f, off_thresh = %f" % (sta, lta, on_thresh, off_thresh))
            
    # Run detector
    detection_list = coincidence_trigger('recstalta', 
                                         on_thresh, 
                                         off_thresh, 
                                         stream,
                                         sta=sta, 
                                         lta=lta,
                                         thr_coincidence_sum=min_chans, 
                                         max_trigger_length=max_trigger_length)
                                                                                  
    print("%d detections" % len(detection_list))
    numsta = list(set([tr.stats.station for tr in stream]))
    
    # Write file
    fname = "detections_sta%3.2f_lta%3.2f_on%2.1f_off%2.1f.list" % (sta, lta, on_thresh, off_thresh)
    with open(fname, "w") as outfile:
        outfile.write("STA = %f, LTA = %f, on_thresh = %f, off_thresh = %f, min_chans = %d, min_sep = %f, max_trigger_length = %f\n" % (sta, lta, on_thresh, off_thresh, min_chans, min_sep, max_trigger_length))
        for detection in detection_list:
            outfile.write("%s\n" % detection["time"].strftime("%Y-%m-%d %H:%M:%S"))
    
    
DATA_DIR_ROOT = "/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz/"
day_folder = "20200310"
stream = Stream()
for sta in ['G2', 'G4', 'G14', 'G6', 'G8', 'G11', 'G16', 'G18']:
    stream += read(os.path.join(DATA_DIR_ROOT, day_folder, "*%s.*.sac" % sta))
print(stream)#.__str__(extended=True))

sta = float(sys.argv[1])
lta = float(sys.argv[2])
on_thresh = float(sys.argv[3])

off_thresh = 0.5*on_thresh
            
print("STA = %f, LTA = %f, on_thresh = %f, off_thresh = %f" % (sta, lta, on_thresh, off_thresh))
            
network_detect(stream, sta, lta, on_thresh)
            
