from obspy import read
import os
import sys
from glob import glob
from obspy.signal.trigger import recursive_sta_lta
from obspy.signal.trigger import coincidence_trigger
from obspy.core.event.catalog import Catalog
from obspy.core.event.event import Event
from obspy.core.event.magnitude import StationMagnitude, Magnitude, StationMagnitudeContribution, Amplitude
from obspy.core.event.base import QuantityError, TimeWindow, WaveformStreamID, Comment
from helper_functions import *
import time
import pandas as pd
import logging
from eqcorrscan.core.match_filter.helpers import _total_microsec
from eqcorrscan.utils.findpeaks import decluster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)

plt.rcParams["figure.figsize"] = [20, 20]
DOM_PERIOD = 1 / 50.
EPS_WINLEN = 1.5 * DOM_PERIOD
DATA_DIR_ROOT = "/home/gilbert_lab/cami_frs/borehole_data/sac_daily_nez_500Hz/"

# TODO: Rotate to RTZ using highest amplitude across 3 channels before picks, get azimuth at that point?


def network_detection(st, cft_return=True):
    # TODO: Dynamic threshold method of Akram 2013

    fs = st[0].stats.sampling_rate
    sta_len_sec = 2.5 * DOM_PERIOD  # 2-3 times dominant period
    lta_len_sec = 7.5 * sta_len_sec  # 5-10 times STA
    nsta = int(sta_len_sec * fs)
    nlta = int(lta_len_sec * fs)
    on_thresh = 3.0  # 3.5
    off_thresh = 0.5
    numsta = len(list(set([tr.stats.station for tr in st])))
    min_chans = numsta*2   # Minimum number of channels to log network detection

    cft_stream = Stream()
    if cft_return:
        for i, tr in enumerate(st.traces):
            cft = recursive_sta_lta(tr.data, nsta=nsta, nlta=nlta)
            # cft = eps_smooth(cft, w=int(EPS_WINLEN * Fs))
            cft_stream += Trace(data=cft, header=tr.stats)
        detection_list = coincidence_trigger(None, on_thresh, off_thresh, cft_stream,
                                             thr_coincidence_sum=min_chans,
                                             max_trigger_length=2.0, delete_long_trigger=True,
                                             details=True)

    else:
        detection_list = coincidence_trigger('recstalta', on_thresh, off_thresh, st,
                                             sta=sta_len_sec, lta=lta_len_sec,
                                             thr_coincidence_sum=min_chans,
                                             max_trigger_length=2.0, delete_long_trigger=True,
                                             details=True)
    # Dictionary keys:
    # time, stations, trace_ids, coincidence_sum, cft_peaks, cft_stds, duration, cft_wmean, cft_std_wmean
    return detection_list, cft_stream


def get_phases(response_curve, idx_search_max, time, pol, fs=500.0, verbose=False):
    # Params
    phase_len_tol = int(10 * DOM_PERIOD * fs)  # assumed length of a phase in samples
    tol = int(8 * DOM_PERIOD * fs)  # buffer to apply before first pick to search for second pick
    search_len = int(1.0 * fs)  # for secondary max
    plan_thresh = 0.7
    rect_thresh = 0.7
    secmax_ratio_thresh = 0.1

    # Get global maximum
    globmax, iglobmax, flag = get_max_diff(response_curve, idx_search_max)
    if not flag:
        if verbose:
            ax = plt.subplot(111)
            t_plt = np.array([mdates.date2num(t._get_datetime()) for t in time])
            ax.plot_date(t_plt, response_curve, "k")
            ax.plot_date(t_plt[idx_search_max], response_curve[idx_search_max], "r")
            ax.set_title("Charcteristic function")
            plt.show()
            # plt.close()
        Logger.warning("Couldn't find the global maximum on the response curve for some reason.... skipping")
        phases = None
        return phases

    tglobmax = time[iglobmax]
    if verbose:
        Logger.info("Global max at %s" % (tglobmax))
    globmax_stats = pol_window_stats(pol, iglobmax, phase_len_tol, show_stats=False)

    # Second max search
    isecmax, secmax, flag2 = get_secondary_max(response_curve, iglobmax, search_len_samp=search_len, tol_samp=tol,
                                               phase_len_samp=phase_len_tol)
    if not flag2:
        if verbose:
            Logger.info("No secondary max found because windows were out of bounds.")
        phS = {"arrival_time": tglobmax,
               "pol_stats": globmax_stats,
               "index": iglobmax}
        phP = {"arrival_time": None,
               "pol_stats": None,
               "index": None}
    else:
        tsecmax = time[isecmax]
        if verbose:
            Logger.info("Secondary max at %s" % (tsecmax))
        secmax_stats = pol_window_stats(pol, isecmax, phase_len_tol, show_stats=False)

        # Decide to keep or reject second maximum and if P or S
        if (secmax_stats["planarity"]["median"] > plan_thresh or secmax_stats["rectilinearity"][
            "median"] > rect_thresh) and secmax > secmax_ratio_thresh * globmax:
            if tsecmax < tglobmax:
                if verbose:
                    Logger.info("Secondary maximum is P-wave")
                phP = {"arrival_time": tsecmax,
                       "pol_stats": secmax_stats,
                       "index": isecmax}
                phS = {"arrival_time": tglobmax,
                       "pol_stats": globmax_stats,
                       "index": iglobmax}
            else:
                if verbose:
                    Logger.info("Secondary maximum is S-wave")
                phS = {"arrival_time": tsecmax,
                       "pol_stats": secmax_stats,
                       "index": isecmax}
                phP = {"arrival_time": tglobmax,
                       "pol_stats": globmax_stats,
                       "index": iglobmax}
        else:
            # Logger.info(
            #     "Rejecting secondary max because under rectilinearity/planarity thresholds: \n\tplanarity = %f\n\trectilinearity = %f" % (
            #         secmax_stats["planarity"]["median"], secmax_stats["rectilinearity"][
            #             "median"]))
            phS = {"arrival_time": tglobmax,
                   "pol_stats": globmax_stats,
                   "index": iglobmax}
            phP = {"arrival_time": None,
                   "pol_stats": None,
                   "index": None}

    phases = {"P": phP, "S": phS}
    return phases


def main(st, fname, verbose=False):
    fs = st[0].stats.sampling_rate

    # Detect STA/LTA for all geodes, with minimum number of stations included
    proc1 = time.time()
    detection_list, cft_stream = network_detection(st, cft_return=True)
    proc2 = time.time()
    Logger.info("Network detection search done in %f s." % (proc2 - proc1))
    Logger.info("Number of network detections = %d" % len(detection_list))

    # Get picks and stats, iterating detection by detection, then station by station
    # Buffer window before and after detection
    buffer1 = 3.0  # 0.2
    buffer2 = 10.0

    # Load ERT data
    ert_surveys_file = "survey_times_ERT.csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    ert_surveys = pd.read_csv(ert_surveys_file, parse_dates=["time_local_start"], date_parser=dateparse)
    ert_surveys["time_local_start"] = ert_surveys["time_local_start"].dt.tz_localize("America/Edmonton",
                                                                                     ambiguous="infer")
    ert_surveys["time_utc_start"] = ert_surveys["time_local_start"].dt.tz_convert(None)
    ert_surveys["time_utc_end"] = ert_surveys["time_utc_start"] + pd.Timedelta(25, unit="m")
    ert_surveys["time_utc_end"] = pd.to_datetime(ert_surveys["time_utc_end"])
    ert_surveys["time_utc_start"] = pd.to_datetime(ert_surveys["time_utc_start"])

    catalog = Catalog()
    # Loop over each STA/LTA detection
    for detection in detection_list:

        # Skip if detection happens during ERT survey
        tmin = detection["time"]._get_datetime()
        is_ert_on = \
            ert_surveys.loc[(ert_surveys['time_utc_start'] <= tmin) & (ert_surveys['time_utc_end'] >= tmin)].shape[
                0] > 0
        if is_ert_on:
            Logger.warning("Skip false detection during ERT survey.")
            continue

        Logger.info("DETECTION TIME: %s\n\t DURATION_SEC: %f" % (detection["time"], detection["duration"]))
        det_start = detection["time"]
        det_end = detection["time"] + detection["duration"]

        # Detection stream
        det_st = st.slice(starttime=det_start - buffer1, endtime=det_end + buffer2)
        det_st.detrend()
        det_st_to_save = det_st.copy()
        t_plt = det_st[0].times("matplotlib")
        t_utc = det_st[0].times("utcdatetime")
        det_cft = cft_stream.slice(starttime=det_start - buffer1, endtime=det_end + buffer2)

        # Stations in detection stream
        station_list = list(set(detection["stations"]))
        station_list.sort()

        # Check if frequencies within window are anomalous
        highf_ratio_threshold = 0.6
        for station in station_list:
            tmp = det_st.select(station=station).copy()
            nbad = 0
            for tr in tmp:
                ratio = highf_ratio(data=tr.data, sampling_rate=fs)
                if ratio > highf_ratio_threshold:
                    nbad += 1
            if nbad > 0:
                for tr in tmp:
                    Logger.warning(
                        "Removing station %s because for %d traces, ratio of frequencies above %f is above %f" %
                        (station, nbad, 0.25 * fs, highf_ratio_threshold))
                    det_st.remove(tr)

        # Stations in detection stream
        station_list = list(set(detection["stations"]))
        station_list.sort()

        if len(station_list) < 4:
            Logger.warning("Only %d stations left, less than 4, so skipping this detection" % len(station_list))

        # Search window for phase around STA/LTA detection time
        idet_start = (np.abs(t_utc - det_start)).argmin()
        idet_end = (np.abs(t_utc - det_end)).argmin()
        idx_search_max = range(idet_start, idet_end)

        # Analyze stations one by one
        pol_st = Stream()
        event_phases = []
        for ista, station in enumerate(station_list):

            # Select waveform and STA-LTA streams
            sta_st = det_st.select(station=station).copy()
            network = sta_st[0].stats.network
            sta_st.detrend()
            sta_cft = det_cft.select(station=station).copy()
            sta_cft_stack = (sta_cft.select(channel="DPZ")[0].data + sta_cft.select(channel="DPN")[0].data +
                             sta_cft.select(channel="DPE")[0].data) / 3

            # Polarization properties
            tpol, pol_dict, pol_st_sta = modified_polarization_analysis(sta_st, dominant_period=DOM_PERIOD,
                                                                        interpolate=True)
            pol_st += pol_st_sta

            # Energy response curve for pick detection
            per = peak_eigenvalue_ratio(pol_dict["eigenvalue1"], win_len=int(2 * DOM_PERIOD * fs))
            per = eps_smooth(per, w=int(EPS_WINLEN * fs))
            jer = joint_energy_ratio(sta_cft_stack, t_plt, per, tpol)

            # Extract phases
            sta_phases = get_phases(response_curve=jer, idx_search_max=idx_search_max, time=t_utc, pol=pol_dict,
                                    verbose=False)
            if sta_phases:

                # Now do some quality control
                snr_threshold = 2.5
                win_len_s = 0.2
                sta_phases["station"] = station
                sta_phases["network"] = network

                if sta_phases["P"]["arrival_time"]:
                    arr_time = sta_phases["P"]["arrival_time"] - 0.02

                    snr, channel = get_snr_phase(sta_st, time=arr_time,
                                                 win_len_s=win_len_s,
                                                 verbose=False,
                                                 tnoise=None)
                    Logger.info(
                        "SNR for P pick %s.%s..%s: %f \t at t = %s" % (network, station, channel, snr, arr_time))
                    if snr < snr_threshold:
                        #Logger.info("P pick below SNR threshold of %f" % snr_threshold)
                        sta_phases["P"]["arrival_time"] = None
                    else:
                        sta_phases["P"]["SNR"] = snr
                        sta_phases["P"]["channel"] = channel

                if sta_phases["S"]["arrival_time"]:
                    arr_time = sta_phases["S"]["arrival_time"] - 0.02
                    if sta_phases["P"]["arrival_time"]:
                        tnoise = sta_phases["P"]["arrival_time"] - 0.02
                    else:
                        tnoise = None
                    snr, channel = get_snr_phase(sta_st.select(), time=arr_time,
                                                 win_len_s=win_len_s,
                                                 verbose=False,
                                                 tnoise=tnoise)

                    Logger.info(
                        "SNR for S pick %s.%s..%s: %f \t at t = %s" % (network, station, channel, snr, arr_time))
                    if snr < snr_threshold:
                        Logger.info("S pick below SNR threshold of %f" % snr_threshold)
                        sta_phases["S"]["arrival_time"] = None
                    else:
                        sta_phases["S"]["SNR"] = snr
                        sta_phases["S"]["channel"] = channel

                Logger.info("Station %s: t_P = %s\tt_S = %s" % (
                    station, sta_phases["P"]["arrival_time"], sta_phases["S"]["arrival_time"]))
                event_phases.append(sta_phases)
            else:
                Logger.info("No phase found for station %s" % station)
            # End of for loop over stations

        if not event_phases:
            Logger.info("No picks found at all for this detection.")
            continue
        else:
            nump = len([p for p in event_phases if p["P"]["arrival_time"]])
            nums = len([p for p in event_phases if p["S"]["arrival_time"]])
            Logger.info("Number of initial picks before MCCC: P = %d, S = %d" % (nump, nums))
        if nump + nums == 0:
            Logger.info("No picks found at all for this detection.")
            continue
        # if verbose:
        #     plot_phases(event_phases, det_st)
        #     wadati_plot(event_phases, det_st)

        # Align with mccc
        Logger.info("Refining picks with MCCC")
        event_phases = align_mccc(event_phases=event_phases, stream=det_st, verbose=False)

        nump = len([p for p in event_phases if p["P"]["arrival_time"]])
        nums = len([p for p in event_phases if p["S"]["arrival_time"]])
        if nump == 0 and nums == 0:
            Logger.warning("No remaining picks after MCCC!")
            continue
        elif nump + nums < 5:
            Logger.info("Less than 5 picks remaining. Skipping event.")
            continue
        if verbose:
            Logger.info("Number of picks after MCCC: P = %d, S = %d" % (nump, nums))
            wadati_plot(event_phases, det_st)
            plot_phases(event_phases, det_st)

        # Update polarization statistics
        Logger.info("Updating polarization attributes")
        phase_len_tol = int(10 * DOM_PERIOD * fs)
        for i, staph in enumerate(event_phases):
            sta_st = det_st.select(station=staph["station"]).copy()
            t = sta_st[0].times("utcdatetime")
            tpol, pol_dict, _ = modified_polarization_analysis(sta_st, dominant_period=DOM_PERIOD, interpolate=True)
            tp = staph["P"]["arrival_time"]
            if tp:
                idxP = np.argmin(np.abs(t - tp))
                stats = pol_window_stats(pol_dict, idxP, phase_len_tol, show_stats=False)
                event_phases[i]["P"]["pol_stats"] = stats
            ts = staph["S"]["arrival_time"]
            if ts:
                idxS = np.argmin(np.abs(t - ts))
                stats = pol_window_stats(pol_dict, idxS, phase_len_tol, show_stats=False)
                event_phases[i]["S"]["pol_stats"] = stats

        # Convert to obspy Picks and Event
        event_picks = []
        for i, staph in enumerate(event_phases):
            event_picks += sta_phases_to_pick(staph=staph)
        event = Event(picks=event_picks)

        # Estimate average event distance using availables pairs of P and S picks
        r_med = distance_from_tstp(event.picks, min_estim=1)
        if not r_med:  # We cannot estimate r, hence magnitude
            Logger.warning("Couldn't estimate hypocentral distance from ts-tp. No magnitude calculation.")
            # Add event to catalog
            if verbose:
                Logger.info("Adding event to catalog: *******************************************")
                Logger.info(event)
            catalog.events.append(event)
            stfilepath = os.path.join("detections_waveforms", det_start.strftime("%Y%m%d"))
            if not os.path.exists(stfilepath):
                os.mkdir(stfilepath)
            det_st_to_save.write(os.path.join(stfilepath, "bhdetect_%s.mseed" % det_start.strftime("%Y%m%d%H%M%S")), format="MSEED")
            
            continue

        # Calculate magnitudes
        Logger.info("Computing magnitudes...")
        magtime_contriblist = []
        magspec_contriblist = []
        for ista, station in enumerate(station_list):
            sta_picks = [p for p in event.picks if p.waveform_id.station_code == station]
            r = distance_from_tstp(sta_picks, min_estim=2)
            if not r:
                r = r_med
            ts = get_pick(event.picks, station, "S")
            if not ts:  # No ts pick
                Logger.warning("There is no S pick for station %s." % station)
                continue
            sta_st = det_st.select(station=station).copy()
            sta_st.detrend()

            # Estimate coda
            tp = get_pick(event.picks, station, "P")
            if not tp:
                tsig = ts - 0.5
            else:
                tsig = tp - 0.02
            tcoda, s_len, snr = get_coda_duration(sta_st.copy(), tsig=tsig, ts=ts, win_len_s=0.2)
            if not tcoda:
                if verbose:
                    Logger.info("Couldn't calculate coda duration for station %s skipping..." % station)
                continue

            # Save coda info
            amp = Amplitude(generic_amplitude=tcoda, snr=snr, type="END", category="duration", unit="s",
                            magnitude_hint="Md")
            event.amplitudes.append(amp)

            # Estimate energy flux
            if tp:
                Logger.info("Calculating energy flux fr station %s" % station)
                epsilonS = 0
                for tr in sta_st.copy():
                    tr_cut = tr.trim(starttime=ts, endtime=ts + (ts - tp)).data
                    cumsum_u2 = scipy.integrate.cumtrapz(tr_cut ** 2, dx=tr.stats.delta)
                    epsilonS += cumsum_u2[-1]
                amp = Amplitude(generic_amplitude=epsilonS, snr=snr, type="A", category="integral",
                                unit="other", time_window=TimeWindow(begin=ts - tp, end=2 * (ts - tp), reference=tp),
                                waveform_id=WaveformStreamID(network_code=tr.stats.network,
                                                             station_code=tr.stats.station))
                event.amplitudes.append(amp)

            # Estimate Mw for each component
            Mw_spec_sta = []
            Mw_time_sta = []
            Q_spec_sta = []
            fc_spec_sta = []
            for tr in sta_st:
                # Cut noise window and S waveform
                noise_len = s_len
                taper_perc = 0.1
                trnoise = tr.copy()
                trnoise.trim(starttime=tsig - (1 + taper_perc) * noise_len, endtime=tsig - taper_perc * noise_len)
                trnoise.taper(type="hann", max_percentage=taper_perc, side="both")
                tr.trim(starttime=ts - taper_perc * s_len, endtime=ts + (1 + taper_perc) * s_len)
                tr.taper(type="hann", max_percentage=taper_perc, side="both")

                # Check SNR
                snr_trace = np.median(tr.slice(starttime=ts, endtime=ts + s_len).data) / \
                            np.median(trnoise.data)

                if snr_trace < 3:
                    Logger.info("SNR < 3, skipping trace for magnitude calculation.")
                    # Poor SNR, skip trace
                    continue

                # Displacement waveform
                trdisp = tr.copy()
                trdisp.integrate()
                trdisp.detrend()

                # Estimate magnitude: time method
                Mw_time, M0_time, omega0_time = estimate_magnitude_time(trdisp, r, disp=False)
                Mw_time_sta.append(Mw_time)

                # Estimate magnitude: spectral method
                Mw_o, M0_o, omega0_o, fc_o, Q_o = estimate_magnitude_spectral(trdisp, r, omega0_time, trnoise=None,
                                                                              disp=False)
                if not Mw_o:
                    Logger.warning("No magnitude found due to errors.")
                    continue
                elif fc_o < 2 or Q_o > 40 or Q_o < 1:  # Qs Attenuation larger than Sandstone=31, shale=10
                    # Reject spectral estimate
                    Logger.warning("Rejecting spectral estimate with: fc = %f, Q = %f" % (fc_o, Q_o))
                    continue
                else:
                    Mw_spec_sta.append(Mw_o)
                    Q_spec_sta.append(Q_o)
                    fc_spec_sta.append(fc_o)

            # Now get average for station as a whole
            Logger.info("Found %d estimates of Mw using time method for station %s." % (len(Mw_time_sta), station))
            Logger.info("Found %d estimates of Mw using spectral method for station %s." % (len(Mw_spec_sta), station))
            if Mw_time_sta:
                smagt = StationMagnitude(mag=np.mean(Mw_time_sta),
                                         mag_errors=QuantityError(uncertainty=np.std(Mw_time_sta)),
                                         station_magnitude_type="Mw_time",
                                         comments=[Comment(text="snr = %f" % snr)]
                                         )
                event.station_magnitudes.append(smagt)
                contrib = StationMagnitudeContribution(station_magnitude_id=smagt.resource_id, weight=snr)
                magtime_contriblist.append(contrib)
                Logger.info("Magnitude time estimate = %f" % np.mean(Mw_time_sta))

            if Mw_spec_sta:
                smags = StationMagnitude(mag=np.mean(Mw_spec_sta),
                                         mag_errors=QuantityError(uncertainty=np.std(Mw_spec_sta)),
                                         station_magnitude_type="Mw_spectral",
                                         comments=[
                                             Comment(text="Q_mean = %f, Q_std = %f" % (
                                                 np.mean(Q_spec_sta), np.std(Q_spec_sta))),
                                             Comment(text="Fc_mean = %f, Fc_std = %f" % (
                                                 np.mean(fc_spec_sta), np.std(fc_spec_sta))),
                                             Comment(text="snr = %f" % snr)]
                                         )
                event.station_magnitudes.append(smags)
                contrib = StationMagnitudeContribution(station_magnitude_id=smags.resource_id, weight=snr)
                magspec_contriblist.append(contrib)
                Logger.info("Magnitude spectral estimate = %f" % np.mean(Mw_spec_sta))
                Logger.info("Fc = %f, Q = %f" % (np.mean(fc_spec_sta), np.mean(Q_spec_sta)))

            # End of for loop over stations

        # Get magnitude for event
        if magspec_contriblist:
            Logger.info("Found %d station estimates of Mw using spectral method." % len(magspec_contriblist))
            wave_num = 0
            wave_den = 0
            val_list = []
            for m in magspec_contriblist:
                mval = [sm.mag for sm in event.station_magnitudes if sm.resource_id == m.station_magnitude_id][0]
                wave_num += mval * m.weight
                wave_den += m.weight
                val_list.append(mval)
            mag = wave_num / wave_den
            mags = Magnitude(mag=mag,
                             mag_errors=np.std(val_list),
                             magnitude_type="Mw_spectral",
                             station_count=len(magspec_contriblist),
                             station_magnitude_contributions=magspec_contriblist)
            event.magnitudes.append(mags)
            Logger.info("Event magnitude estimate using spectral method: Mw = %f" % mags.mag)
        if magtime_contriblist:
            Logger.info("Found %d station estimates of Mw using time method." % len(magtime_contriblist))
            wave_num = 0
            wave_den = 0
            val_list = []
            for m in magtime_contriblist:
                mval = [sm.mag for sm in event.station_magnitudes if sm.resource_id == m.station_magnitude_id][0]
                wave_num += mval * m.weight
                wave_den += m.weight
                val_list.append(mval)
            mag = wave_num / wave_den
            magt = Magnitude(mag=mag,
                             mag_errors=np.std(val_list),
                             magnitude_type="Mw_time",
                             station_count=len(magtime_contriblist),
                             station_magnitude_contributions=magtime_contriblist)
            event.magnitudes.append(magt)
            Logger.info("Event magnitude estimate using time method: Mw = %f" % magt.mag)

        # Add event to catalog
        if verbose:
            Logger.info("Adding event to catalog: *******************************************")
            Logger.info(event)
        catalog.events.append(event)
        stfilepath = os.path.join("detections_waveforms", det_start.strftime("%Y%m%d"))
        if not os.path.exists(stfilepath):
            os.mkdir(stfilepath)
            det_st_to_save.write(os.path.join(stfilepath, "bhdetect_%s.mseed" % det_start.strftime("%Y%m%d%H%M%S")), format="MSEED")

    if len(catalog) > 0:
        # Decluster
        declustered_catalog = decluster_bh(catalog, trig_int=2.0)
        if not os.path.exists(os.path.split(fname)[0]):
            os.mkdir(os.path.split(fname)[0])
        declustered_catalog.write(fname, format="QUAKEML")
        # catalog.write(fname, format="QUAKEML")


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
        cat_fname = "detections/%s/bhdetections_%s_hour%02d.xml" % (daystr, daystr, hour)
        # if os.path.exists(cat_fname):
        #     Logger.warning("Hour is already processed, skipping.")
        #     continue

        # Read data
        stream = read(os.path.join(wf_dir, "*.sac"), starttime=starttime, endtime=endtime)

        # Check for bad data:

        blacklist_stations = ["G4", "G5", "G16"]
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
