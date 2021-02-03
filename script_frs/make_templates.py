from glob import glob
from template_gen_modified import *
import sys
import time
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from eqcorrscan.core.match_filter.helpers import _total_microsec
from eqcorrscan.utils.findpeaks import decluster
from eqcorrscan.core.match_filter import Party, Tribe
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)


DETECT_DIR = "/home/gilbert_lab/cami_frs/borehole_data/energy_detector/detections_clean"
WF_DIR_ROOT = "/home/gilbert_lab/cami_frs/sac_hourly_125Hz/"
OUTPUT_DIR = "/home/gilbert_lab/cami_frs/borehole_data/energy_detector/initial_templates_daily"
SAMPLING_RATE = 125.0


def do_fft(signal, delta):
    """Compute the complex Fourier transform of a signal."""
    npts = len(signal)
    # if npts is even, we make it odd
    # so that we do not have a negative frequency in the last point
    # (see numpy.fft.rfft doc)
    if not npts % 2:
        npts -= 1

    fft = np.fft.rfft(signal, n=npts) * delta
    fftfreq = np.fft.fftfreq(len(signal), d=delta)
    fftfreq = fftfreq[0:fft.size]
    return np.abs(fft), fftfreq, fft


def highf_ratio(data):
    ndt = 1/SAMPLING_RATE
    power, f, _ = do_fft(data, ndt)
    fnyq4 = 0.25 * SAMPLING_RATE  # half of Nyquist frequency (1/4 sampling rate)
    energy_highf = np.sum(power[f > fnyq4])/np.sum(power)
    return energy_highf


def get_most_common_stations(cat, num=3):
    from collections import Counter

    id_lst = []
    for ev in cat:
        for p in ev.picks:
            id_str = "%s" % (p.waveform_id.station_code)
            id_lst.append(id_str)
    c = Counter(id_lst)
    top3_stations = [item[0] for item in c.most_common(num)]
    return top3_stations


def decluster_bh(cat, trig_int=2.0):
    detect_info = []
    all_detections = []
    for ev in cat:
        all_detections.append(ev)
        tpicks = [p.time for p in ev.picks]
        detect_time = min(tpicks)
        detect_val = len(ev.picks)
        detect_info.append((detect_time, detect_val))
    # Now call decluster
    min_det = sorted([d[0] for d in detect_info])[0]
    detect_vals = np.array([d[1] for d in detect_info], dtype=np.float32)
    detect_times = np.array([_total_microsec(d[0].datetime, min_det.datetime) for d in detect_info])
    peaks_out = decluster(
        peaks=detect_vals, index=detect_times,
        trig_int=trig_int * 10 ** 6)
    # Need to match both the time and the detection value
    declustered_catalog = Catalog()
    for ind in peaks_out:
        matching_time_indices = np.where(detect_times == ind[-1])[0]
        matches = matching_time_indices[
            np.where(detect_vals[matching_time_indices] == ind[0])[0][0]]
        declustered_catalog.append(all_detections[matches])

    return declustered_catalog


def get_event_waveforms(cat, wf_dir="/home/gilbert_lab/cami_frs/sac_hourly_125Hz/", wf_len=3.0, selection=None,
                        plotting=False, saving=False):
    ndt = 1 / 125.0
    # wf_len = 2.0
    pre_pick = int((1.0) / ndt) * ndt
    post_pick = int((wf_len - 1.0) / ndt) * ndt

    template_list = []
    for ev in cat.events:
        print(ev)
        ptimes = [p.time for p in ev.picks if p.phase_hint == "P"]
        stimes = [p.time for p in ev.picks if p.phase_hint == "S"]
        arrtimes = stimes
        tmin = min([t for t in arrtimes if t])
        tmax = max([t for t in arrtimes if t])
        tmed = tmin + (tmax - tmin) * 0.5
        # pattern = os.path.join(wf_dir, "G*", tmin.strftime("BH.G*..DP*%Y-%m-%d_%H*2020*"))
        if selection:
            flist = []
            for sta in selection:
                pattern = os.path.join(wf_dir, sta, tmin.strftime("*..DP*%Y-%m-%d_%H*2020*"))
                flist += glob(pattern)
        else:
            pattern = os.path.join(wf_dir, "*", tmin.strftime("*..DP*%Y-%m-%d_%H*2020*"))
            flist = glob(pattern)

        if len(flist) == 0:
            print(pattern)
            print("no waveform data found")
            continue
        st = Stream()
        for file in flist:
            st += read(file, starttime=tmin - pre_pick, endtime=tmin + post_pick)
        st.resample(125.0)
        st.detrend("demean")
        st.filter("lowpass", freq=60)
        st.detrend("demean")
        print("starttime: %s" % tmin)

        # Keep a subset
        if selection:
            for tr in st:
                if tr.stats.station not in selection:  # in ["G4","G5","G6","G8","G11", "G12","G13", "G14", "G15","G16","G17"]:
                    st.remove(tr)

                    # Add to template list
        tup = (st, ev.resource_id)
        template_list.append(tup)

        if saving:
            if len(ptimes) > 2:  # and len(stimes) >5:
                tmin = min(ptimes)
                tmax = max(stimes)
                fname = "event_%s" % tmin.strftime("%Y%m%d%H%M%S")
                ev.write(fname + ".xml", format="QUAKEML")
                st.slice(starttime=tmin - pre_pick, endtime=tmax + post_pick).write(fname + ".mseed", format="MSEED")
                print("Saved this event")
                print(ev)

        # Plot
        if plotting:
            stations = list(set([p.waveform_id.station_code for p in ev.picks]))
            fig, axs = plt.subplots(len(stations), 3, sharex=True, figsize=(15, 3 * len(stations)))
            for i, sta in enumerate(stations):
                tp = [p.time for p in ev.picks if p.waveform_id.station_code == sta and p.phase_hint == "P"]
                ts = [p.time for p in ev.picks if p.waveform_id.station_code == sta and p.phase_hint == "S"]
                tt = []
                tts = []
                if tp:
                    tt.append(tp[0])
                if ts:
                    tt.append(ts[0])
                    tts.append(ts[0])
                tmin = min(tts)
                tmax = max(tts)
                sta_st = st.select(station=sta).slice(starttime=tmin - pre_pick, endtime=tmax + post_pick)
                tplt = sta_st[0].times("matplotlib")
                tpicks = [mdates.date2num(t._get_datetime()) for t in tt]
                axs[i][0].plot_date(tplt, sta_st.select(channel="DPN")[0], "k")
                axs[i][0].vlines(tpicks, min(axs[i][0].get_ylim()), max(axs[i][0].get_ylim()), color="r")
                axs[i][1].plot_date(tplt, sta_st.select(channel="DPE")[0], "k")
                axs[i][1].vlines(tpicks, min(axs[i][1].get_ylim()), max(axs[i][1].get_ylim()), color="r")
                axs[i][2].plot_date(tplt, sta_st.select(channel="DPZ")[0], "k")
                axs[i][2].vlines(tpicks, min(axs[i][2].get_ylim()), max(axs[i][2].get_ylim()), color="r")
                axs[i][0].set_title(highf_ratio(sta_st.select(channel="DPN")[0].data, ndt))
                axs[i][1].set_title(highf_ratio(sta_st.select(channel="DPE")[0].data, ndt))
                axs[i][2].set_title(highf_ratio(sta_st.select(channel="DPZ")[0].data, ndt))
            plt.show()
            plt.close()

    return template_list


def get_catalog(daystr):

    # Get borehole detections
    flist = glob(os.path.join(DETECT_DIR, "bhdetections_%s_hour*.xml" % (daystr)))
    full_cat = Catalog()
    flist.sort()
    for file in flist:
        Logger.info("Reading detections from %s" % os.path.split(file)[1])
        tmp = read_events(file)
        full_cat += tmp
    Logger.info("Total number of BH detections: %d" % len(full_cat))

    # Fetch other catalog files
    other_cat_files = glob("/home/gilbert_lab/cami_frs/eqcorrscan/templates_hawksOnly_f60/detections*/*%s*.tgz" % daystr)
    other_cat_files += glob("/home/gilbert_lab/cami_frs/eqcorrscan/templates_wBH_f60/detections*/*%s*.tgz" % daystr)

    # Look for new borehole detections only, not accounted for in previous EQcorrscan runs
    matched_cat = Catalog()
    matched_detect = []
    parties = []
    for file in other_cat_files:
        Logger.info("Looking for matching detections in previous EQcorrscan run in file:\n\t%s" % file)
        party = Party().read(file)
        parties.append((file, party))
        for f in party:
            for d in f.detections:
                for ev in full_cat:
                    tmin = min([p.time for p in ev.picks])
                    if abs(tmin - d.detect_time) < 2.0:
                        matched_cat.append(ev)
    Logger.info("%d matched detections" % len(matched_cat))
    cat = Catalog()
    for ev in full_cat:
        if ev not in matched_cat:
            cat += ev
    Logger.info("Total number of NEW BH detections: %d" % len(cat))

    # new_cat_wp = Catalog()
    # for ev in cat:
    #     if ev not in matched_cat and len([p for p in ev.picks if p.phase_hint == "P"]) > 0 and len(ev.magnitudes) > 0:
    #         new_cat_wp += ev
    # print(new_cat_wp)
    # print("Total number of NEW BH detections with P picks and magnitudes: %d" % len(new_cat_wp))

    return cat


def construct_tribe(catalog, stream):
    # parameters for templates
    ndt = 1 / SAMPLING_RATE
    prepick = int(0.05 / ndt) * ndt # this ensures all templates have same length
    length = int(0.5 / ndt) * ndt  # this ensures all templates have same length
    lowcut = None
    highcut = 60.0
    filt_order = 3
    process_len = stream[0].stats.endtime - stream[0].stats.starttime
    min_snr = 3
    num_cores = cpu_count()
    templates, catalog, process_lengths = custom_template_gen(method='from_meta_file',
                                                              meta_file=catalog,
                                                              process_len=process_len,
                                                              st=stream,
                                                              lowcut=lowcut,
                                                              highcut=highcut,
                                                              samp_rate=SAMPLING_RATE,
                                                              filt_order=filt_order,
                                                              length=length,
                                                              prepick=prepick,
                                                              swin='all',
                                                              all_horiz=False,
                                                              min_snr=min_snr,
                                                              num_cores=num_cores,
                                                              ignore_bad_data=True,
                                                              plot=False, return_event=True)

    from eqcorrscan.core.match_filter import Template
    from obspy.core.event import Comment, CreationInfo
    template_list = []
    for template, event, process_len in zip(templates, catalog,
                                            process_lengths):
        if len(template) < 4:
            Logger.warning("Less than 4 traces for this template. Skipping. Number of picks was %d" % len(event.picks))
            continue
        t = Template()
        for tr in template:
            if not np.any(tr.data.astype(np.float16)):
                Logger.warning('Data are zero in float16, missing data,'
                               ' will not use: {0}'.format(tr.id))
                template.remove(tr)
        if len(template) == 0:
            Logger.error('Empty template. Skipping')
            continue        
            
        # Check if traces have same length
        lengths = set([tr.stats.endtime - tr.stats.starttime for tr in template])
        if len(lengths) > 1:
            print("Traces don't have same lengths. Fixing")
            wf_len = list(lengths)[0]
            for tr in template:        
                tr.trim(starttime=tr.stats.starttime, endtime=tr.stats.starttime + wf_len, fill_value=0, pad=True)        
            
        t.st = template
        t.name = template.sort(['starttime'])[0]. \
            stats.starttime.strftime('%Y_%m_%dt%H_%M_%S')
        t.lowcut = lowcut
        t.highcut = highcut
        t.filt_order = filt_order
        t.samp_rate = SAMPLING_RATE
        t.process_length = process_len
        t.prepick = prepick
        event.comments.append(Comment(
            text="eqcorrscan_template_" + t.name))
        t.event = event
        template_list.append(t)
    tribe = Tribe(templates=template_list)

    return tribe


def catalog_to_templates(catalog):

    tribe = Tribe()
    for hour in range(0, 24):
        pattern = os.path.join(WF_DIR_ROOT, "G*", "*..DP*%s-%s-%s_%02d*2020*" % (year, month, day, hour))
        if not glob(pattern):
            Logger.info("No data found for hour %d" % hour)
            continue
        st_hr = read(pattern)
        st_hr.resample(SAMPLING_RATE)
        st_hr.detrend()

        starttime = st_hr[0].stats.starttime
        endtime = st_hr[0].stats.endtime
        cat_hr = Catalog()
        for ev in catalog:
            if starttime < ev.picks[0].time < endtime:
                cat_hr += ev
        Logger.info("Processing %d detections during hour %d" % (len(cat_hr), hour))

        # For each pick find the channel with highest amplitude
        cat_template = Catalog()
        for iev, ev in enumerate(cat_hr):
            nbad = 0
            for ip, p in enumerate(ev.picks):
                sta = p.waveform_id.station_code
                cut_st = st_hr.select(station=sta).slice(starttime=p.time-0.02, endtime=p.time + 0.1)
                imax = np.argmax(cut_st.max())
                ev.picks[ip].waveform_id.channel_code = cut_st[imax].stats.channel
                ratio = highf_ratio(data=cut_st[imax].data)
                if ratio > 0.75:
                    nbad += 1
            if nbad > 3:
                Logger.info("Removing event with ratio of high frequencies = %f" % ratio)
            elif len(ev.picks) > 4:
                cat_template += ev

        # Make a tribe for this hour
        if len(cat_template) > 0:
            tribe_hour = construct_tribe(catalog=cat_template, stream=st_hr)
            tribe += tribe_hour
            Logger.info("Adding: %s" % tribe_hour)

    Logger.info("Initial tribe for this whole day: %s" % tribe)
    return tribe


if __name__ == "__main__":

    daystr = sys.argv[1] # "20200310"
    year = daystr[0:4]
    month = daystr[4:6]
    day = daystr[6:]

    # Get catalog of new borehole events
    cat = get_catalog(daystr=daystr)
    Logger.info("Finished loading catalog.")

    # Create tribe
    tribe = catalog_to_templates(catalog=cat)
    Logger.info("Finished tribe construction.")
    print(tribe)
    tribe_fname = os.path.join(OUTPUT_DIR, "tribe_init_day%s.tgz" % daystr)
    Logger.info("Saving initial tribe to: %s" % tribe_fname)    
    tribe.write(filename=tribe_fname)
    
    # Detect:
    trig_int = 2.0
    min_chans = 5
    threshold = 0.7
    threshold_type = "av_chan_corr"
    parties = Party()
    for hour in range(0, 24):
        Logger.info("Starting detection for hour %d" % hour)
        pattern = os.path.join(WF_DIR_ROOT, "G*", "*..DP*%s-%s-%s_%02d*2020*" % (year, month, day, hour))
        if not glob(pattern):
            Logger.info("No data found for hour %d" % hour)
            continue
        st = read(pattern)
        st.resample(SAMPLING_RATE)
        st.detrend()
        party = tribe.detect(stream=st, threshold=threshold, threshold_type=threshold_type, trig_int=trig_int,
                             plot=False, daylong=False, ignore_bad_data=True,
                             parallel_proces=True, cores=cpu_count(), concurrency="multiprocess",
                             group_size=20, overlap="calculate")
        party.min_chans(min_chans)
        #party.decluster(trig_int=trig_int)
        party = party.filter(dates=[st[0].stats.starttime, st[0].stats.endtime], min_dets=2)
        parties += party

    # Save party
    template_list = [f.template for f in parties if f]
    
    if len(template_list) > 0:  
        party_fname = os.path.join(OUTPUT_DIR, "party_day%s.tgz" % daystr)
        Logger.info("Saving final party to: %s" % party_fname)
        parties.write(filename=party_fname)
        # Save tribe
        final_tribe = Tribe(templates=template_list)
        tribe_fname = os.path.join(OUTPUT_DIR, "tribe_day%s.tgz" % daystr)
        Logger.info("Saving final tribe to: %s" % tribe_fname)    
        final_tribe.write(filename=tribe_fname)
    
    
