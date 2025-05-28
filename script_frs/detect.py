from obspy import read
import os
import sys
from glob import glob
from detector.helper_functions import *
from detector.detector import *
from obspy import read
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

OUTPUT_DIR = "/home/genevieve.savard/borehole-energy-detector/script_frs/detections"

if __name__ == "__main__":

    datestr = sys.argv[1]
    year = int(datestr[0:4])
    month = int(datestr[4:6])
    day = int(datestr[6:])

    wf_dir = os.path.join(DATA_DIR_ROOT, datestr)
    flist = glob(os.path.join(wf_dir, "*.sac"))
    Logger.info("Number of files for this day = %d" % len(flist))
    if len(flist) == 0:
        sys.exit()
    for hour in range(0, 24):

        starttime = UTCDateTime(year, month, day, hour, 0, 0)
        endtime = starttime + 3600
        Logger.info("Processing hour %d: from %s to %s" % (hour, starttime, endtime))
        daystr = "%d%02d%02d" % (year, month, day)
        cat_fname = os.path.join(OUTPUT_DIR, "%s/bhdetections_%s_hour%02d.xml" % (daystr, daystr, hour))
        # if os.path.exists(cat_fname):
        #     Logger.warning("Hour is already processed, skipping.")
        #     continue

        # Read data
        stream = read(os.path.join(wf_dir, "*.sac"), starttime=starttime, endtime=endtime)

        # Check for bad data:
        blacklist_stations = ["BH004", "BH005", "BH016"]
        for tr in stream:
            mmax = np.max(tr.data)
            if np.isnan(mmax):
                Logger.warning("Bad trace: %s" % tr)
                stream.remove(tr)
            elif tr.stats.station in blacklist_stations:
                stream.remove(tr)

        if len(stream) > 0:

            stream.detrend("demean")
            stream.detrend("linear")

            # check that sampling rates do not vary
            fs = stream[0].stats.sampling_rate
            if len(stream) != len(stream.select(sampling_rate=fs)):
                msg = "sampling rates of traces in stream are not equal"
                raise ValueError(msg)

            # Check data for gaps
            if stream.get_gaps():
                msg = 'Input stream must not include gaps:\n' + str(stream)
                raise ValueError(msg)

            # Run detector
            main(stream, cat_fname, verbose=True)
        else:
            Logger.warning("No data for this hour")
            continue

    # Test window:
    # starttime = UTCDateTime(2020, 3, 10, 6, 57, 40)
    # endtime = starttime + 30
