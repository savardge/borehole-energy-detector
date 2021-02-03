from eqcorrscan.core.match_filter import Tribe
from obspy import read, Catalog
import matplotlib.pyplot as plt
import math
from glob import glob
import os
import sys

DETECT_DIR = "/home/gilbert_lab/cami_frs/borehole_data/energy_detector/detections_clean"
WF_DIR_ROOT_500Hz = "/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz"
WF_DIR_ROOT_HAWK = "/home/gilbert_lab/cami_frs/hawk_data/sac_data_raw/"
party_dir = "/home/gilbert_lab/cami_frs/borehole_data/energy_detector/initial_templates_daily_old/"
out_dir = "/home/gilbert_lab/cami_frs/borehole_data/energy_detector/templates"


def get_full_template(template, prepick=4.0, length=8.0):
    st = template.st
    tmin = min([tr.stats.starttime for tr in st])
    ndt = 1/500.0
    wf_len_s = int(length / ndt) * ndt   
    pattern = os.path.join(WF_DIR_ROOT_500Hz, tmin.strftime("%Y%m%d"), "*DP*")
    detst = read(pattern, starttime=tmin - prepick, endtime=tmin - prepick + wf_len_s)
    pattern_hawk = os.path.join(WF_DIR_ROOT_HAWK, "*", "*", "*DP*.D.%s*" % tmin.strftime("%Y%m%d"))
    if glob(pattern_hawk):
        ndt = 1 / 250.0
        wf_len_s = int(8.0 / ndt) * ndt
        prepick = 4.0
        detst += read(pattern_hawk, starttime=tmin - prepick, endtime=tmin - prepick + wf_len_s)
    detst.detrend("demean")
    return detst


if __name__ == "__main__":

    daystr = sys.argv[1]
    print("************ %s ************" % daystr)
    tribe = Tribe().read(os.path.join(party_dir, "tribe_day%s.tgz" % daystr))
    print(tribe)
    if not tribe.templates:
        print("no templates...")
        sys.exit()

    for template in tribe:
        if not template:
            continue
        fname = os.path.join(out_dir, "%s_full_template.mseed" % template.name)
        if os.path.exists(fname):
            continue
        num_p_picks = len([p for p in template.event.picks if p.phase_hint == "P"])
        has_magn = len(template.event.magnitudes) > 0
        if len(template.st) > 5 and (num_p_picks > 0 or has_magn):
            print(template.name)
            full_template_st = get_full_template(template)
            full_template_st.write(fname, format="MSEED")
            print("Saving %s" % fname)
            fname = fname.replace("full_template.mseed", "template.tgz")
            template.write(fname)
            print("Saving %s" % fname)
