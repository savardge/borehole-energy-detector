import os
from obspy import read, UTCDateTime, read_events
import sys
from glob import glob

# Input
timestr = sys.argv[1]
buffer_before = 10
buffer_after = 20
wfdir = "/home/gilbert_lab/cami_frs/all_daily_symlinks/"

# Then save waveform
torig = UTCDateTime(timestr)
tstart = torig - buffer_before
tend = torig + buffer_after
evdir = os.path.join("/home/genevieve.savard/borehole-energy-detector/script_frs", "waveforms", "event_%s" % torig.strftime("%Y%m%d%H%M%S"))
if not os.path.exists(evdir):
    os.mkdir(evdir)
# List of daily waveforms files top extract from
wflist = glob(os.path.join(wfdir, torig.strftime("%Y/%j"), "*"))

for file in wflist:

    st = read(file, starttime=tstart, endtime=tend)
    if not st:
        continue
    tr = st[0]
    station = tr.stats.station
    network = tr.stats.network
    channel = tr.stats.channel
    starttime = tr.stats.starttime.strftime("%Y%m%d%H%M%S")
    endtime = tr.stats.endtime.strftime("%Y%m%d%H%M%S")
    outfile = os.path.join(evdir, "%s.%s..%s.%s_%s.mseed" % (network, station, channel, starttime, endtime))
    if os.path.exists(outfile):
        continue

    print("Saving waveform to %s" % outfile)
    st.write(outfile, format="MSEED")


# First save event
#good_catalog = read_events("good_catalogue.xml", format="QuakeML")
#event = good_catalog[idx]
#obs_file = "/home/genevieve.savard/cami/catalog/events/event_%s.obs" % torig.strftime("%Y%m%d%H%M%S")
#if os.path.exists(obs_file):
#    print(obs_file)
#    print("Event already processed. Exiting.")
#    sys.exit()

#torig = event.origins[0].time
#event.write(obs_file, "NLLOC_OBS")

