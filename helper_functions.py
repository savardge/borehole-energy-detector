from obspy import UTCDateTime

from obspy.signal.polarization import _get_s_point
from obspy.signal.invsim import cosine_taper
from obspy import Stream, Trace
import math
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import warnings
import mtspec
import scipy
import logging
import numpy as np
from pyrocko.util import str_to_time
from pyrocko.gui.marker import PhaseMarker

Logger = logging.getLogger(__name__)

# params for magnitude
VP = 1800.  # m/s
VS = 630.  # m/s
DENSITY = 2600.  # Shale-sandstone: 2000-2600 kg/m3
radpat = 0.63  # Average for S-waves
SPECTRAL_MODEL = "brune"  # "boatwright


# Utilities
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


def highf_ratio(data, sampling_rate):
    ndt = 1 / sampling_rate
    power, f, _ = do_fft(data, ndt)
    fnyq4 = 0.25 * sampling_rate  # half of Nyquist frequency (1/4 sampling rate)
    energy_highf = np.sum(power[f > fnyq4]) / np.sum(power)
    return energy_highf


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # Logger.info(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y[int(window_len / 2 - 1):-int(window_len / 2 + 1)]


def sliding_window(a, len=10, step=1, copy=False):
    """
    Input:
    a: array input
    L: Window len
    S: Stride len/stepsize
    Output:
    returns a 2D array, each row being a window
    """
    nrows = ((a.size - len) // step) + 1
    n = a.strides[0]
    view = np.lib.stride_tricks.as_strided(a, shape=(nrows, len), strides=(step * n, n))
    if copy:
        return view.copy()
    else:
        return view


def eps_smooth(sig, w=5):
    """
    Edge Preserving Smoothing using window length of n samples (Luo et al. 2002)
    Sabbione and Velis (2010) recommend a filter length equal to one and half signal periods
    """
    n = sig.shape[0]
    wins = sliding_window(sig, len=w, step=1, copy=False)
    wins_std = wins.std(axis=1)
    wins_mean = wins.mean(axis=1)
    idx = sliding_window(wins_std, len=w, step=1).argmin(axis=1) + np.arange(0, n - 2 * (w - 1))
    sig_eps = np.pad(wins_mean[idx], w - 1, mode='constant')
    return sig_eps


def mccc(seis, dt, twin, ccmin, comp='Z'):
    from scipy.linalg import lstsq
    """ FUNCTION [TDEL,RMEAN,SIGR] = MCCC(SEIS,DT,TWIN);
    Function MCCC determines optimum relative delay times for a set of seismograms based on the
    VanDecar & Crosson multi-channel cross-correlation algorithm. SEIS is the set of seismograms.
    It is assumed that this set includes the window of interest and nothing more since we calculate the
    correlation functions in the Fourier domain. DT is the sample interval and TWIN is the window about
    zero in which the maximum search is performed (if TWIN is not specified, the search is performed over
    the entire correlation interval).
    APP added the ccmin, such that only signals that meet some threshold similarity contribute to the delay times. """

    # Set nt to twice length of seismogram section to avoid
    # spectral contamination/overlap. Note we assume that
    # columns enumerate time samples, and rows enumerate stations.
    # Note in typical application ns is not number of stations...its really number of events
    # all data is from one station
    nt = np.shape(seis)[1] * 2
    ns = np.shape(seis)[0]
    tcc = np.zeros([ns, ns])

    # Copy seis for normalization correction
    seis2 = np.copy(seis)

    # Set width of window around 0 time to search for maximum
    # mask = np.ones([1,nt])
    # if nargin == 3:
    itw = int(np.fix(twin / (2 * dt)))
    mask = np.zeros([1, nt])[0]
    mask[0:itw + 1] = 1.0
    mask[nt - itw:nt] = 1.0

    # Zero array for sigt and list on non-zero channels
    sigt = np.zeros(ns)

    # First remove means, compute autocorrelations, and find non-zeroed stations.
    for iss in range(0, ns):
        seis[iss, :] = seis[iss, :] - np.mean(seis[iss, :])
        ffiss = np.fft.fft(seis[iss, :], nt)
        acf = np.real(np.fft.ifft(ffiss * np.conj(ffiss), nt))
        sigt[iss] = np.sqrt(max(acf))

    # Determine relative delay times between all pairs of traces.
    r = np.zeros([ns, ns])
    tcc = np.zeros([ns, ns])

    # Two-Channel normalization ---------------------------------------------------------

    # This loop gets a correct r by checking how many channels are actually being compared
    if comp == 'NE':

        # First find the zero-channels (the np.any tool will fill in zeroNE)
        # zeroNE ends up with [1,0], [0,1], or [1,1] for each channel, 1 meaning there IS data
        zeroNE = np.zeros([ns, 2])
        dum = np.any(seis2[:, 0:nt / 4], 1, zeroNE[:, 0])
        dum = np.any(seis2[:, nt / 4:nt / 2], 1, zeroNE[:, 1])

        # Now start main (outer) loop
        for iss in range(0, ns - 1):
            ffiss = np.conj(np.fft.fft(seis[iss, :], nt))

            for jss in range(iss + 1, ns):

                ffjss = np.fft.fft(seis[jss, :], nt)
                # ccf  = np.real(np.fft.ifft(ffiss*ffjss,nt))*mask
                ccf = np.fft.fftshift(np.real(np.fft.ifft(ffiss * ffjss, nt)) * mask)
                cmax = np.max(ccf)

                # chcor for channel correction sqrt[ abs( diff[jss] - diff[iss]) + 1]
                # This would be perfect correction if N,E channels always had equal power, but for now is approximate
                chcor = np.sqrt(abs(zeroNE[iss, 0] - zeroNE[jss, 0] - zeroNE[iss, 1] + zeroNE[jss, 1]) + 1)

                # OLD, INCORRECT chcor
                # chcor = np.sqrt( np.sum(zeroNE[iss,:])+np.sum(zeroNE[jss,:]) - (zeroNE[iss,0]*zeroNE[jss,0]+zeroNE[iss,1]*zeroNE[jss,1]) )

                rtemp = cmax * chcor / (sigt[iss] * sigt[jss])

                # Quadratic interpolation for optimal time (only if CC found > ccmin)
                if rtemp > ccmin:

                    ttemp = np.argmax(ccf)

                    x = np.array(ccf[ttemp - 1:ttemp + 2])
                    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])

                    [a, b, c] = lstsq(A, x)[0]

                    # Solve dy/dx = 2ax + b = 0 for time (x)
                    tcc[iss, jss] = -b / (2 * a) + ttemp

                    # Estimate cross-correlation coefficient
                    # r[iss,jss] = cmax/(sigt[iss]*sigt[jss])
                    r[iss, jss] = rtemp
                else:
                    tcc[iss, jss] = nt / 2

                    # Reguar Normalization Version -------------------------------------------------------
    elif comp != 'NE':
        for iss in range(0, ns - 1):
            ffiss = np.conj(np.fft.fft(seis[iss, :], nt))
            for jss in range(iss + 1, ns):

                ffjss = np.fft.fft(seis[jss, :], nt)
                # ccf  = np.real(np.fft.ifft(ffiss*ffjss,nt))*mask
                ccf = np.fft.fftshift(np.real(np.fft.ifft(ffiss * ffjss, nt)) * mask)
                cmax = np.max(ccf)

                rtemp = cmax / (sigt[iss] * sigt[jss])

                # Quadratic interpolation for optimal time (only if CC found > ccmin)
                if rtemp > ccmin:

                    ttemp = np.argmax(ccf)

                    x = np.array(ccf[ttemp - 1:ttemp + 2])
                    A = np.array([[1, -1, 1], [0, 0, 1], [1, 1, 1]])

                    [a, b, c] = lstsq(A, x)[0]

                    # Solve dy/dx = 2ax + b = 0 for time (x)
                    tcc[iss, jss] = -b / (2 * a) + ttemp

                    # Estimate cross-correlation coefficient
                    # r[iss,jss] = cmax/(sigt[iss]*sigt[jss])
                    r[iss, jss] = rtemp
                else:
                    tcc[iss, jss] = nt / 2

                    #######################################################

    # Some r could have been made > 1 due to approximation, fix this
    r[r >= 1] = 0.99

    # Fisher's transform of cross-correlation coefficients to produce
    # normally distributed quantity on which Gaussian statistics
    # may be computed and then inverse transformed
    z = 0.5 * np.log((1 + r) / (1 - r))
    zmean = np.zeros(ns)
    for iss in range(0, ns):
        zmean[iss] = (np.sum(z[iss, :]) + np.sum(z[:, iss])) / (ns - 1)
    rmean = (np.exp(2 * zmean) - 1) / (np.exp(2 * zmean) + 1)

    # Correct negative delays (for fftshifted times)
    # ix = np.where( tcc>nt/2);  tcc[ix] = tcc[ix]-nt
    tcc = tcc - nt / 2

    # Subtract 1 to account for sample 1 at 0 lag (Not in python)
    # tcc = tcc-1

    # Multiply by sample rate
    tcc = tcc * dt

    # Use sum rule to assemble optimal delay times with zero mean
    tdel = np.zeros(ns)

    # I changed the tdel calculation to not include zeroed-out waveform pairs in normalization
    for iss in range(0, ns):
        ttemp = np.append(tcc[iss, iss + 1:ns], -tcc[0:iss, iss])
        tdel[iss] = np.sum(ttemp) / (np.count_nonzero(ttemp) + 1)
        # tdel[iss] = ( np.sum(tcc[iss,iss+1:ns])-np.sum(tcc[0:iss,iss]) )/ns

    # Compute associated residuals
    res = np.zeros([ns, ns])
    sigr = np.zeros(ns)
    for iss in range(0, ns - 1):
        for jss in range(iss + 1, ns):
            res[iss, jss] = tcc[iss, jss] - (tdel[iss] - tdel[jss])

    for iss in range(0, ns):
        sigr[iss] = np.sqrt((np.sum(res[iss, iss + 1:ns] ** 2) + np.sum(res[0:iss, iss] ** 2)) / (ns - 2))

    return tdel, rmean, sigr, r, tcc


def get_median_filtered(picks, threshold=3):
    ts = [p["S"]["arrival_time"].timestamp for p in picks if p["S"]["arrival_time"]]

    differenceS = np.abs(ts - np.median(ts))
    median_differenceS = np.median(differenceS)

    tp = [p["P"]["arrival_time"].timestamp for p in picks if p["P"]["arrival_time"]]
    differenceP = np.abs(tp - np.median(tp))
    median_differenceP = np.median(differenceP)

    for i, p in enumerate(picks):
        if median_differenceS > 0 and p["S"]["arrival_time"] and np.abs(
                p["S"]["arrival_time"] - np.median(ts)) / float(median_differenceS) > threshold:
            picks[i]["S"]["arrival_time"] = None
            Logger.info("Removed S pick at station %s" % p["station"])
        if median_differenceP > 0 and p["P"]["arrival_time"] and np.abs(
                p["P"]["arrival_time"] - np.median(tp)) / float(median_differenceP) > threshold:
            picks[i]["P"]["arrival_time"] = None
            Logger.info("Removed P pick at station %s" % p["station"])
    return picks


def align_mccc(event_phases, stream, verbose=False):
    from obspy import Stream
    from collections import Counter

    cc_threshold = 0.65  # min CC coefficient to accept pick
    pre_pick = 0.2  # pre pick window in seconds
    post_pick = 0.5  # post pick window in seconds
    min_sep_sec = 0.2  # Min separation in seconds between P and S

    fs = stream[0].stats.sampling_rate
    streamp = Stream()
    streams = Stream()
    maxchansp = []
    maxchanss = []
    s_ptimes = [d["S"]["arrival_time"] - d["P"]["arrival_time"] for d in event_phases if
                d["P"]["arrival_time"] and d["S"]["arrival_time"]]
    s_ptime_median = max([np.median(s_ptimes), pre_pick])

    for i, staph in enumerate(event_phases):
        station = staph["station"]
        tp = staph["P"]["arrival_time"]
        ts = staph["S"]["arrival_time"]
        if tp and ts:
            dum = stream.select(station=station).copy().trim(starttime=tp - pre_pick, endtime=tp + s_ptime_median,
                                                             pad=True, fill_value=0, nearest_sample=False)
            maxchansp.append(get_channel_max_amplitude(dum))
            dum.normalize()
            dum.taper(max_percentage=0.2)
            streamp += dum
        elif tp and not ts:
            dum = stream.select(station=station).copy().trim(starttime=tp - pre_pick, endtime=tp + post_pick,
                                                             nearest_sample=False)
            maxchanss.append(get_channel_max_amplitude(dum))
            dum.normalize()
            streams += dum
        if ts:
            dum = stream.select(station=station).copy().trim(starttime=ts - pre_pick, endtime=ts + post_pick,
                                                             nearest_sample=False)
            maxchanss.append(get_channel_max_amplitude(dum))
            dum.normalize()
            streams += dum

    if len(streamp) > 2:
        chan = Counter(maxchansp).most_common(1)[0][0][0]
        Logger.info("Channel for P: %s " % chan)
        vlen = min([tr.data.shape[0] for tr in streamp.select(channel=chan)])
        plist = [tr.data[:vlen] for tr in streamp.select(channel=chan)]
        stalst = [tr.stats.station for tr in streamp.select(channel=chan)]
        seisp = np.vstack(plist)
        tdelp, rmeanp, sigrp, rp, tccp = mccc(seisp, 1 / fs, s_ptime_median, cc_threshold)
        if verbose:
            for sta, t, r in zip(stalst, tdelp, rmeanp):
                Logger.info("%s: %f, %f" % (sta, t, r))
        for i, staph in enumerate(event_phases):
            if staph["P"]["arrival_time"] and staph["station"] in stalst:
                idx = stalst.index(staph["station"])
                if tdelp[idx] == 0:  # and (rmeanp[idx] == 0 or math.isnan(rmeanp[idx])):
                    event_phases[i]["P"]["arrival_time"] = None
                else:
                    event_phases[i]["P"]["arrival_time"] = staph["P"]["arrival_time"] - tdelp[idx]
    if len(streams) > 2:
        chan = Counter(maxchanss).most_common(1)[0][0][0]
        Logger.info("Channel for S: %s " % chan)
        vlen = min([tr.data.shape[0] for tr in streams.select(channel=chan)])
        slist = [tr.data[:vlen] for tr in streams.select(channel=chan)]
        stalst = [tr.stats.station for tr in streams.select(channel=chan)]
        seiss = np.vstack(slist)
        tdels, rmeans, sigrs, rs, tccs = mccc(seiss, 1 / fs, post_pick, cc_threshold)
        if verbose:
            for sta, t, r in zip(stalst, tdels, rmeans):
                Logger.info("%s: %f, %f" % (sta, t, r))
        for i, staph in enumerate(event_phases):
            if staph["S"]["arrival_time"] and staph["station"] in stalst:

                idx = stalst.index(staph["station"])
                if tdels[idx] == 0:  # and (rmeans[idx] == 0 or math.isnan(rmeans[idx])):
                    event_phases[i]["S"]["arrival_time"] = None
                else:
                    event_phases[i]["S"]["arrival_time"] = staph["S"]["arrival_time"] - tdels[idx]

    # Now check if new P arrival is within 0.2 s of S arrival
    for i, staph in enumerate(event_phases):
        if staph["P"]["arrival_time"] and staph["S"]["arrival_time"] and staph["S"]["arrival_time"] - staph["P"][
            "arrival_time"] < min_sep_sec:
            event_phases[i]["P"]["arrival_time"] = None

    return event_phases


def get_pick(picks, station, phase, time_only=True):
    pp = [p for p in picks if p.waveform_id.station_code == station and p.phase_hint == phase]
    pick = None
    if len(pp) == 1:
        pick = pp[0]
    elif len(pp) > 1:
        Logger.warning(
            "There was for than one pick found for station %s and phase %s. Keeping only the first found." % (
                station, phase))
        pick = pp[0]
    if time_only and pick:
        out = pick.time
    else:
        out = pick
    return out


def get_channel_max_amplitude(stream):
    mmax = 0
    chan = None
    for tr in stream:
        if max(tr.data) > mmax:
            mmax = max(tr.data)
            chan = tr.stats.channel
    return chan, mmax


# Polarization

def modified_flinn(stream, noise_thresh=0):
    mask = (stream[0][:] ** 2 + stream[1][:] ** 2 + stream[2][:] ** 2) > noise_thresh
    x = np.zeros((3, mask.sum()), dtype=np.float64)
    # East
    x[0, :] = stream[2][mask]
    # North
    x[1, :] = stream[1][mask]
    # Z
    x[2, :] = stream[0][mask]

    covmat = np.cov(x)
    eigvec, eigenval, v = np.linalg.svd(covmat)
    # Rectilinearity defined after Montalbetti & Kanasewich, 1970
    rect = 1.0 - np.sqrt(eigenval[1] / eigenval[0])
    # Planarity defined after [Jurkevics1988]_
    plan = 1.0 - (2.0 * eigenval[2] / (eigenval[1] + eigenval[0]))
    azimuth = math.degrees(math.atan2(eigvec[0][0], eigvec[1][0]))
    eve = np.sqrt(eigvec[0][0] ** 2 + eigvec[1][0] ** 2)
    incidence = math.degrees(math.atan2(eve, eigvec[2][0]))
    if azimuth < 0.0:
        azimuth = 360.0 + azimuth
    if incidence < 0.0:
        incidence += 180.0
    if incidence > 90.0:
        incidence = 180.0 - incidence
        if azimuth > 180.0:
            azimuth -= 180.0
        else:
            azimuth += 180.0
    if azimuth > 180.0:
        azimuth -= 180.0

    return azimuth, incidence, rect, plan, eigenval[0], eigenval[1], eigenval[2]


def modified_polarization_analysis(stream, dominant_period=0.02, interpolate=False):
    win_len = 4 * dominant_period  # 2 x dominant period
    win_frac = 0.2
    res = []
    stime = stream[0].stats.starttime
    etime = stream[0].stats.endtime
    fs = stream[0].stats.sampling_rate

    if len(stream) != 3:
        msg = 'Input stream expected to be three components:\n' + str(stream)
        raise ValueError(msg)

    spoint, _epoint = _get_s_point(stream, stime, etime)

    nsamp = int(win_len * fs)
    nstep = int(nsamp * win_frac)
    newstart = stime
    tap = cosine_taper(nsamp, p=0.22)
    offset = 0
    while (newstart + (nsamp + nstep) / fs) < etime:
        try:
            for i, tr in enumerate(stream):
                dat = tr.data[spoint[i] + offset:
                              spoint[i] + offset + nsamp]
                dat = (dat - dat.mean()) * tap
                if tr.stats.channel[-1].upper() == "Z":
                    z = dat.copy()
                elif tr.stats.channel[-1].upper() == "N":
                    n = dat.copy()
                elif tr.stats.channel[-1].upper() == "E":
                    e = dat.copy()
                else:
                    msg = "Unexpected channel code '%s'" % tr.stats.channel
                    raise ValueError(msg)

            data = [z, n, e]
        except IndexError:
            break

        # we plot against the centre of the sliding window
        azimuth, incidence, reclin, plan, eigenval1, eigenval2, eigenval3 = modified_flinn(data)
        res.append(np.array([newstart.timestamp + float(nstep) / fs,
                             azimuth, incidence, reclin, plan, eigenval1, eigenval2, eigenval3]))

        offset += nstep
        newstart += float(nstep) / fs

    res = np.array(res)
    result_dict = {
        "timestamp": res[:, 0],
        "azimuth": res[:, 1],
        "incidence": res[:, 2],
        "rectilinearity": res[:, 3],
        "planarity": res[:, 4],
        "eigenvalue1": res[:, 5],
        "eigenvalue2": res[:, 6],
        "eigenvalue3": res[:, 7],
    }
    time1 = [UTCDateTime(dum) + win_len for dum in result_dict["timestamp"]]
    time = [mdates.date2num(dum._get_datetime()) for dum in time1]
    delta = time1[1] - time1[0]

    pol_st = Stream()
    for key in ["azimuth", "incidence", "rectilinearity", "planarity"]:
        data = result_dict[key]
        header = {"delta": delta, "network": stream[0].stats.network,
                  "station": stream[0].stats.station,
                  "location": stream[0].stats.location,
                  "channel": key[0:3].upper(), "starttime": time1[0],
                  "endtime": time1[-1]}
        pol_st += Trace(data=data, header=header)

    if interpolate:
        t_trace = stream[0].times("matplotlib")
        for key in ["azimuth", "incidence", "rectilinearity", "planarity", "eigenvalue1", "eigenvalue2", "eigenvalue3"]:
            v = result_dict[key]
            vi = np.interp(t_trace, time, v)
            result_dict[key] = vi
        time = t_trace
    return time, result_dict, pol_st


# Energy ratio


def peak_eigenvalue_ratio(eig, win_len=1):
    """
    eig: array of principal eigenvalue
    win_len: window length in samples. Should be 2-3 times dominant period
    """
    vmax = sliding_window(eig, len=win_len, step=1).max(axis=1)
    wbefore = vmax[:-(win_len + 1)]
    wafter = vmax[win_len + 1:]
    # per = np.pad(wafter/wbefore, win_len, mode='empty')
    per = np.pad(wafter / wbefore, win_len, mode='constant')
    return per


def joint_energy_ratio(cft, tcft, per, tper):
    """ Compute Joint Energy Ratio (Akram 2013)"""
    # Interpolate PER to time array for sta/lta
    peri = np.interp(tcft, tper, per)
    # Normalize
    jer = (peri / np.max(peri)) * (cft / np.max(cft))
    return jer


def ipick_diff(diff, on_trigger, pre_pick=50):
    istart_pick = on_trigger - pre_pick
    iend_pick = on_trigger
    return np.argmax(diff[istart_pick:iend_pick]) + istart_pick


def get_trigger_vec(on_off, shape):
    flag = False
    on_off_vec = np.array(on_off)
    trigger_vec = np.zeros(shape)
    if on_off.any() and on_off.shape[0] > 0:
        flag = True
        for row in on_off:  # range(0, on_off.shape[1]):
            on_off_vec[row[0]:row[1]] += 1  # for the stack
            trigger_vec[row[0]:row[1]] = 1

    return trigger_vec, flag


def get_max_diff(data, idx, w=50, verbose=False):
    """Finds the maximum in the data within the indices specified by array idx
    then calculates the maximum of the derivative within a window w before the data maximum."""

    if verbose:
        Logger.info("get_max_diff: idx[0] = %d, idx[-1] = %d" % (idx[0], idx[-1]))

    success = True
    data_diff = np.diff(data, n=1, append=0)

    # Look for max in data within indices idx
    dcut = data[idx]
    val = np.max(dcut)
    ix = np.argmax(dcut) + idx[0]

    # Evaluate max from first derivative within a window w preceding data max
    if ix - w < 0 or ix == len(data):  # Sanity check: ix within bounds
        Logger.warning("Maximum found at edge of array...")
        success = False
        val = None
        imax = None
    else:
        # vald = np.max(dcut_diff[ix - w:ix])
        imax = np.argmax(data_diff[ix - w:ix]) + ix - w

    return val, imax, success


def get_secondary_max(response_curve, iglob, search_len_samp, tol_samp, phase_len_samp, before_only=True):
    success = True

    # window before global max
    wb_start = max([0, iglob - tol_samp - search_len_samp])
    wb_end = max([0, iglob - tol_samp])
    if wb_end > 0:  # Check that window is not entirely before first sample
        idxb = range(wb_start, wb_end)
        maxb, imaxb, flagb = get_max_diff(response_curve, idxb)
    else:
        maxb = 0
        imaxb = None

    # window after global max
    wa_start = min([iglob + phase_len_samp, len(response_curve)])
    wa_end = min([iglob + phase_len_samp + search_len_samp, len(response_curve)])
    if wa_start < len(response_curve):
        idxa = range(wa_start, wa_end)
        maxa, imaxa, flaga = get_max_diff(response_curve, idxa)
    else:
        maxa = 0
        imaxa = None

    # Now decide which one to keep
    if not maxa or not maxb:  # One of the values is None
        success = False
        isecmax = None
        secmax = None
    elif maxa == 0 and maxb == 0:  # Both values are zero
        success = False
        isecmax = None
        secmax = None
    elif before_only:
        isecmax = imaxb
        secmax = maxb
    elif maxb > maxa and not before_only:  # peak in window before higher
        isecmax = imaxb
        secmax = maxb
    elif not before_only:  # peak in window after higher
        isecmax = imaxa
        secmax = maxa

    return isecmax, secmax, success


def pol_window_stats(pol, ix, win_len_samp, show_stats=False):
    azim = {"mean": np.mean(pol["azimuth"][ix:ix + win_len_samp]),
            "median": np.median(pol["azimuth"][ix:ix + win_len_samp]),
            "std": np.std(pol["azimuth"][ix:ix + win_len_samp])}
    plan = {"mean": np.mean(pol["planarity"][ix:ix + win_len_samp]),
            "median": np.median(pol["planarity"][ix:ix + win_len_samp]),
            "std": np.std(pol["planarity"][ix:ix + win_len_samp])}
    rect = {"mean": np.mean(pol["rectilinearity"][ix:ix + win_len_samp]),
            "median": np.median(pol["rectilinearity"][ix:ix + win_len_samp]),
            "std": np.std(pol["rectilinearity"][ix:ix + win_len_samp])}
    inc = {"mean": np.mean(pol["incidence"][ix:ix + win_len_samp]),
           "median": np.median(pol["incidence"][ix:ix + win_len_samp]),
           "std": np.std(pol["incidence"][ix:ix + win_len_samp])}
    stats = {"rectilinearity": rect, "planarity": plan, "azimuth": azim, "incidence": inc}

    if show_stats:
        Logger.info("Azimuth:        mean = %f\tmedian = %f\tstd = %f" % (azim["mean"], azim["median"], azim["std"]))
        Logger.info("Planarity:      mean = %f\tmedian = %f\tstd = %f" % (plan["mean"], plan["median"], plan["std"]))
        Logger.info("Rectilinearity: mean = %f\tmedian = %f\tstd = %f" % (rect["mean"], rect["median"], rect["std"]))
        Logger.info("Incidence: mean = %f\tmedian = %f\tstd = %f" % (inc["mean"], inc["median"], inc["std"]))

    return stats


# SNR

def snr_1c(trace, tsig, sig_len_s, tnoise, noise_len_s):
    data = trace.data
    fs = trace.stats.sampling_rate
    time_array = trace.times("utcdatetime")

    isig = np.argmin(np.abs(time_array - tsig))
    inoise = np.argmin(np.abs(time_array - tnoise))

    sig_len_samp = int(sig_len_s * fs)
    noise_len_samp = int(noise_len_s * fs)
    sig = np.sqrt(np.mean(np.abs(data[isig:isig + sig_len_samp]) ** 2))

    # Max signal
    sigmax = max(np.abs(data[isig:isig + sig_len_samp]))

    # Noise RMS
    noise = np.sqrt(np.mean(np.abs(data[inoise:inoise + noise_len_samp]) ** 2))

    # ax = plt.subplot(111)
    # ax.plot_date(trace.times("matplotlib"), trace.data, "k")
    # for t in [tsig, tsig+sig]
    #     ax.vlines(t, min(trace.data), max(trace.data), color="r")

    return sig / noise


def snr_3c(stream, tsig, sig_len, tnoise, noise_len):
    t = stream[0].times("utcdatetime")
    fs = stream[0].stats.sampling_rate
    isig = int(np.where(t == tsig)[0][0])
    inoise = int(np.where(t == tnoise)[0][0])

    comp1 = stream.select(channel="DPZ")[0].data
    comp2 = stream.select(channel="DPN")[0].data
    comp3 = stream.select(channel="DPE")[0].data
    dat = np.abs(comp1) + np.abs(comp2) + np.abs(comp3)

    sig_len_samp = int(sig_len * fs)
    noise_len_samp = int(noise_len * fs)
    sig = np.sqrt(np.mean(dat[isig:isig + sig_len_samp] ** 2))
    sigmax = max(dat[isig:isig + sig_len_samp])

    noise = np.sqrt(np.mean(dat[inoise:inoise + noise_len] ** 2))

    return sigmax / noise


def get_snr_phase(st, time, win_len_s, verbose=False, tnoise=None):
    tsig = time
    if not tnoise:
        tnoise = time - 1.5 * win_len_s
    channels = [tr.stats.channel for tr in st]
    if verbose:
        Logger.info("Checking SNR for each pick.")
        fig, axs = plt.subplots(3, 1, sharex=True)
        snr_chan = []
        for channel, ax in zip(channels, axs):
            tr = st.select(channel=channel)[0].copy()
            ax.plot_date(tr.times("matplotlib"), tr.data, "k")
            for t, color in zip([tsig, tsig + win_len_s, tnoise, tnoise + win_len_s], ["r", "r", "b", "b"]):
                ax.vlines(mdates.date2num(t._get_datetime()), min(tr.data), max(tr.data), color=color)
            snr = snr_1c(trace=tr, tsig=tsig, sig_len_s=win_len_s,
                         tnoise=tnoise, noise_len_s=win_len_s)
            ax.set_title("SNR = %f" % snr)
            snr_chan.append((channel, snr))
            Logger.info("%s: snr = %f" % (channel, snr))
        plt.show()
    else:
        snr_chan = []
        for channel in channels:
            tr = st.select(channel=channel)[0].copy()
            snr = snr_1c(trace=tr, tsig=tsig, sig_len_s=win_len_s,
                         tnoise=tnoise, noise_len_s=win_len_s)
            snr_chan.append((channel, snr))

    # Get highest snr
    snr_chan.sort(key=lambda x: x[1], reverse=True)
    # print(snr_chan)
    snr_max = [tup[1] for tup in snr_chan][0]
    best_channel = [tup[0] for tup in snr_chan][0]
    # if tr.stats.station == "C0909":
    #     st.fiddle()
    return snr_max, best_channel


def snr_ratio(data, time_array, tsig, win_len, tnoise=None):
    isig = (np.abs(time_array - tsig)).argmin()
    if not tnoise:
        inoise = isig - win_len
    else:
        inoise = (np.abs(time_array - tnoise)).argmin()
    vstd = sliding_window(data, len=win_len, step=1).std(axis=1)
    noise = np.mean(vstd[:inoise])
    snr = np.pad(vstd / noise, [win_len - 1, 0], mode='edge')
    return snr


# Spectral analysis

def fft_spectrum(stream):
    spec = []
    ndt = stream[0].stats.delta
    fs = stream[0].stats.sampling_rate
    siglen = stream[0].data.shape[0]

    # Calculate spectrum for each component
    for tr in stream:
        Pxx, f = do_fft(tr.data, ndt)
        spec.append(Pxx)

    # Combine 3 components
    specu = np.transpose(np.vstack(spec))
    spec3c = np.sqrt(np.sum(np.abs(specu) ** 2, axis=1))

    return spec3c, f


def multitaper_spectrum(stream):
    import mtspec

    if len(stream) > 3:
        raise ValueError("stream has more than 3 components!")

    spec = []
    ndt = stream[0].stats.delta
    fs = stream[0].stats.sampling_rate
    siglen = stream[0].data.shape[0]

    # Calculate spectrum for each component
    for tr in stream:
        siglen = tr.data.shape[0]
        nfft = scipy.fft.next_fast_len(tr.data.shape[0])
        Pxx, f = mtspec.multitaper.mtspec(tr.data, ndt, 3, nfft=nfft)
        spec.append(Pxx)
        Pxx[1:-2] = Pxx[1:-2] * 0.5
        spec.append(np.sqrt(Pxx * (siglen * fs)) / float(siglen))

    # Combine 3 components
    specu = np.transpose(np.vstack(spec))
    spec3c = np.sqrt(np.sum(np.abs(specu) ** 2, axis=1))

    spec3c = spec[0]

    ix = np.where(f > 0)
    f = f[ix]
    spec3c = spec3c[ix]
    return spec3c, f


def get_freqcorner(spectrum, fv):
    from scipy.signal import medfilt

    # spectrum[np.where(fv < 10)] = 0;

    spec0 = medfilt(spectrum, kernel_size=5)  # 11
    imax = np.argmax(spec0)
    fc = fv[imax]
    return fc


def getOmega0(dspec, f, fc0, Q, T):
    import scipy
    ik = np.where(f < 60)  # keep only frequencies below 60 Hz
    dspec = dspec[ik]
    f = f[ik]

    ik = np.where(f > 10)  # keep only frequencies below 60 Hz
    dspec = dspec[ik]
    f = f[ik]

    # fc0 = f[1]
    omega0_init = dspec[np.where(f == fc0)][0]
    x0 = [omega0_init, fc0]

    # Optimization search
    #     Logger.info("initial x: ")
    #     Logger.info(x0)
    bestx = scipy.optimize.fmin(magnitude_spectrum_func, x0, args=(f, dspec, T),
                                maxiter=10000, maxfun=10000, disp=False)
    omega0 = bestx[0]
    fc = bestx[1]

    Logger.info("omega0 best = %e" % omega0)
    Logger.info("omega0 init = %e" % omega0_init)
    Logger.info("Best corner f = %f" % fc)

    # mspec_best = magnitude_spectrum_model(omega0, fc, T, f)
    # ax = plt.subplot(111)
    # ax.loglog(f, dspec)
    # ax.loglog(f, mspec_best, "r")
    # ax.hlines([omega0, omega0_init], min(f), max(f), color=["green", "black"], linestyle="--")
    # ax.set_title("Observed and modeled spectra")
    # ax.vlines(fc0, min(dspec), max(dspec), color="purple", linestyle="--")
    # plt.show()

    return omega0, fc


def calculate_magnitude_german(det_st, event_phases, do_plot=False):
    # Parameters
    VP = 1800.
    VS = 630.  # m/s
    DENSITY = 2600.
    Q = 65.
    radpat = 0.63

    pre_pick = 0.1
    post_pick = 0.5

    Omega0list = []
    fclist = []
    M0list = []
    Qlist = []
    freqlist = np.empty((len(event_phases), 2048))
    dspeclist = np.empty((len(event_phases), 2048))
    vspeclist = np.empty((len(event_phases), 2048))

    if do_plot:
        fig, axs = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(15, 8))
        ax1 = axs[0]
        ax1.set_title("Velocity")
        # ax1.set_xlim(left=1)
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("spectrum")
        ax1.grid(which="both", axis="both")
        ax2 = axs[1]
        ax2.set_title("Displacement")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Displacement spectrum")
        ax2.grid(which="both", axis="both")

    for i, staph in enumerate(event_phases):
        station = staph["station"]

        tp = staph["P"]["arrival_time"]
        ts = staph["S"]["arrival_time"]
        if tp and ts:
            r = (ts - tp) / (1 / VS - 1 / VP)
        else:
            r = 500.
        Logger.info("r = %f" % r)

        if ts:
            stream = det_st.select(station=staph["station"]).copy().trim(starttime=ts - pre_pick,
                                                                         endtime=ts + post_pick)
            stream.detrend()
            stream.taper(type="hann", max_percentage=0.05, side="both")

            #             vspec, f = fft_spectrum(stream)
            vspec, f = multitaper_spectrum(stream)
            Logger.info(vspec.shape)
            vspeclist[i, :] = vspec

            # Get initial corner frequency from Velocity spectrum
            fc0 = get_freqcorner(vspec, f)

            # Displacement spectrum
            stdisp = stream.copy()
            stdisp.integrate()
            stdisp.detrend()
            stdisp.taper(type="hann", max_percentage=0.2, side="both")
            #             dspec, f = fft_spectrum(stdisp)
            dspec, f = multitaper_spectrum(stdisp)
            dspeclist[i, :] = dspec
            freqlist[i, :] = f

            # Spectral inversion for Omega0 and fc
            T = r / VS
            Logger.info("T = %f" % T)
            omega0, fc = getOmega0(dspec, f, fc0, Q, T)
            m0 = (4 * np.pi * DENSITY * (VS ** 3) * omega0 * r) / radpat
            Omega0list.append(omega0)
            fclist.append(fc)
            M0list.append(m0)
            # Logger.info("Station %s, Omega0 = %f, Fc = %f" % (station, omega0, fc))

            if do_plot:
                T = r / VS

                modeled_mspec = omega0 * np.exp(- np.pi * f * T / Q) / np.sqrt(1 + (f / fc) ** 4)
                ax1.loglog(f[1:], vspec[1:], color="gray")
                # ax1.vlines([fc0, fc], min(vspec), max(vspec), color=["b", "r"], linestyle="--")

                ax2.loglog(f[1:], dspec[1:], color="gray")
                ax2.loglog(f, modeled_mspec, "r")
                # ix = np.where(f > 1)
                # ax1.set_ylim((min(vspec[ix]), max(vspec[ix])))
                # ax2.set_ylim((min(dspec[ix]), max(dspec[ix])))

    if do_plot:
        plt.show()
        # plt.close()

    fcorner = np.median(fclist)
    M0 = np.median(Omega0list)
    Mw = (2. / 3.) * (np.log10(M0) - 9.1)

    if do_plot:
        plt.show()
        # plt.close()

    return Mw, M0, fcorner


# Magnitude

def magnitude_spectrum_model(Omega0, fc, T, Q, f, disp=False):
    model = "brune"  # "boatwright
    #     Q = 65
    if disp:
        Logger.info("omega0 = %f" % Omega0)
        Logger.info("fc = %e" % fc)
        Logger.info("T = %e" % T)
        Logger.info("f.shape = %f" % f.shape)

    if model == "brune":
        mspec = Omega0 * np.exp(- np.pi * f * T / Q) / np.sqrt(1 + (f / fc) ** 4)  # Brune model
    elif model == "boatwright":
        mspec = Omega0 * np.exp(- np.pi * f * T / Q) / (1 + (f / fc) ** 2)  # Boatwright model
    else:
        raise ValueError("Unrecognized model name: choose \"brune\" or \"boatwright\"")

    return mspec


def magnitude_spectrum_func(x, f, spec, T):
    fc, Omega0, Q = x
    # Logger.info("fc = %f, omega = %f, Q= %f" % (fc, Omega0, Q))
    if SPECTRAL_MODEL == "brune":
        mspec = Omega0 * np.exp(- np.pi * f * T / Q) / np.sqrt(1 + (f / fc) ** 4)  # Brune model
    elif SPECTRAL_MODEL == "boatwright":
        mspec = Omega0 * np.exp(- np.pi * f * T / Q) / (1 + (f / fc) ** 2)  # Boatwright model
    else:
        raise ValueError("Unrecognized model name: choose \"brune\" or \"boatwright\"")

    # Misfit function:
    func = np.sum((spec - mspec) ** 2)

    return func


def estimate_magnitude_spectral(trdisp, r, omega0_time, trnoise=None, disp=True):
    # Normalization: https://www.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html

    ndt = trdisp.stats.delta
    Fs = 1 / ndt

    # Frequency band to fit
    min_freq = 10
    max_freq = 60

    # Compute observed displacement spectrum
    datad = trdisp.data
    siglen = datad.shape[0]
    nfft = scipy.fft.next_fast_len(siglen)
    Psig, fd = mtspec.multitaper.mtspec(datad, ndt, 3, nfft=nfft)
    specd = np.sqrt(0.5 * Psig * (siglen * Fs)) / siglen

    if len(specd) < 21 or len(np.argwhere(~np.isfinite(specd))) > 0:
        Logger.warning("spectrum has non-finite values.")
        Mw_o = None
        M0_o = None
        omega0_o = None
        fc_o = None
        Q_o = None
        return Mw_o, M0_o, omega0_o, fc_o, Q_o
    else:
        specd = smooth(specd, window_len=21, window="bartlett")

    if trnoise:
        # Compute observed displacement spectrum
        datan = trnoise.data
        siglen = datan.shape[0]
        nfft = scipy.fft.next_fast_len(siglen)
        Pnoise, fn = mtspec.multitaper.mtspec(datan, ndt, 3, nfft=nfft)
        specnoise = np.sqrt(0.5 * Pnoise * (siglen * Fs)) / siglen
        specnoise = smooth(specnoise, window_len=21, window="bartlett")
        # Subtract noise spectrum
        f, noise_ind, sig_ind = np.intersect1d(specnoise, specd, return_indices=True)
        specd[sig_ind] = specd[sig_ind] - specnoise[noise_ind]
    else:
        f = fd

    # Optimization parameters
    T = r / VS
    fc_try = slice(2, 60, 0.5)
    dom = (omega0_time * 100 - omega0_time / 100) / 50
    omega0_try = slice(omega0_time / 100, omega0_time * 100, dom)
    Q_try = slice(10, 40, 5)
    rranges = (fc_try, omega0_try, Q_try)

    # Cut spectrum to bandpass of interest for fit
    ix = np.where(f > min_freq)
    f_cut = f[ix]
    specd_cut = specd[ix]
    ix = np.where(f_cut < max_freq)
    f_cut = f_cut[ix]
    specd_cut = specd_cut[ix]

    # Optimization
    params = (f_cut, specd_cut, T)
    x0 = scipy.optimize.brute(magnitude_spectrum_func, rranges, args=params,
                              full_output=False, disp=False, finish=scipy.optimize.fmin)
    fc_o = x0[0]
    omega0_o = x0[1]
    Q_o = x0[2]

    # Calculate Mw
    M0_o = (4 * np.pi * DENSITY * (VS ** 3) * omega0_o * r) / radpat
    Mw_o = (2. / 3.) * (np.log10(M0_o) - 9.1)

    if disp:
        Logger.info("Spectral estimate: \n\tomega0 = %e\n\tM0 = %e\n\tMw = %f" % (omega0_o, M0_o, Mw_o))
        Logger.info("Spectral params fitted: \n\tFc = %f\n\tQ = %f" % (fc_o, Q_o))

        # Model spectrum from optimized parameters
        specm = magnitude_spectrum_model(omega0_o, fc_o, T, Q_o, f)

        # Plot
        ax = plt.subplot(111)
        #ax.set_title("Displacement")
        ax.set_title("%s: Mw = %.2f, Fc = %.1f, Q = %d" % (trdisp.id, Mw_o, fc_o, Q_o))
        ax.loglog(fd, specd, "k")
        ax.loglog(f_cut, specd_cut, "b")
        ax.loglog(f, specm, "r")
        ax.set_xlim(left=1)
        ax.set_ylim(bottom=min(specd), top=10*max(specd))
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Displacement spectrum")
        ax.grid(which="both", axis="both")
        plt.show()
        plt.pause(1)
        plt.close()

    return Mw_o, M0_o, omega0_o, fc_o, Q_o


def estimate_magnitude_time(trdisp, r, disp=True):
    datad = trdisp.data
    omega0_time = np.sum(np.abs(datad))
    m0_time = 4 * np.pi * VS ** 3 * r * omega0_time / radpat
    mw_time = (2. / 3.) * (np.log10(m0_time) - 9.1)

    if disp:
        Logger.info("Time estimate: \n\tomega0 = %e\n\tM0 = %e\n\tMw = %f" % (omega0_time, m0_time, mw_time))

    return mw_time, m0_time, omega0_time


def distance_from_tstp(picks, min_estim=1):
    r = None
    station_list = list(set([p.waveform_id.station_code for p in picks]))
    rlist = []
    for station in station_list:
        tp = get_pick(picks, station, "P")
        ts = get_pick(picks, station, "S")
        if tp and ts:
            rlist.append((ts - tp) / (1 / VS - 1 / VP))
    if len(rlist) >= min_estim:
        r = np.median(rlist)
    elif len(rlist) > 0:
        Logger.info("Estimated values of r: ")
        Logger.info(rlist)
        Logger.warning("Not enough picks (found %d) to estimate r" % len(rlist))
    else:
        Logger.warning("No (tp, ts) pairs found to estimate r")
    return r


def get_coda_duration(st, tsig, ts, win_len_s):
    from obspy.signal.trigger import trigger_onset
    tcoda = None
    s_len = None
    snr = None
    ndt = st[0].stats.delta
    win_len = int(win_len_s / ndt)
    data3c = np.empty(st[0].data.shape)
    for tr in st:
        data3c += tr.data
    snr_norm = snr_ratio(data3c, tr.times("utcdatetime"), tsig, win_len)
    if snr_norm.shape[0] == 0:
        Logger.error("something wrong happened while in function snr_ratio()")
        return tcoda, s_len, snr

    ixs = (np.abs(tr.times("utcdatetime") - ts)).argmin()
    thr_on = 0.7 * max(snr_norm[ixs + 1:ixs + 1 + win_len])
    thr_off = 1.0
    if thr_on <= thr_off:
        Logger.info("Thr_on %f <= thr_off %f" % (thr_on, thr_off))
        Logger.info("Trying new threshold:")
        thr_on = 0.7 * max(snr_norm[ixs + 1:])
        if thr_on <= thr_off:
            Logger.error("Still couldn't find a threshold above noise.")
            return tcoda, s_len, snr
    try:
        on_off = trigger_onset(snr_norm[ixs:], thr_on, thr_off)
    except:
        Logger.error("Something wrong during trigger_onset function call.")
        return tcoda, s_len, snr
    else:
        if len(on_off) > 0 and on_off[0][1] > 10:
            snr = np.max(snr_norm[ixs:ixs + on_off[0][1]])
            t_off = ts + on_off[0][1] * ndt
            tcoda = t_off - tsig
            s_len = t_off - ts
        else:
            Logger.error("Trigger_onset couldn't find any trigger")

    return tcoda, s_len, snr


# To QuakeML format

def sta_phases_to_pick(staph):
    from obspy.core.event.base import WaveformStreamID
    from obspy.core.event.origin import Pick

    picks = []
    if staph["P"]["arrival_time"]:
        wf_id = WaveformStreamID(network_code=staph["network"],
                                 location_code="",
                                 station_code=staph["station"],
                                 channel_code=staph["P"]["channel"])
        comments = pol_stats_to_comments(staph["P"]["pol_stats"])
        p = Pick(phase_hint="P", time=staph["P"]["arrival_time"],
                 waveform_id=wf_id,
                 comments=comments)
        picks.append(p)
    if staph["S"]["arrival_time"]:
        wf_id = WaveformStreamID(network_code=staph["network"],
                                 location_code="",
                                 station_code=staph["station"],
                                 channel_code=staph["S"]["channel"])
        comments = pol_stats_to_comments(staph["S"]["pol_stats"])
        p = Pick(phase_hint="S", time=staph["S"]["arrival_time"],
                 waveform_id=wf_id,
                 comments=comments)
        picks.append(p)
    return picks


def pol_stats_to_comments(pol_stats):
    from obspy.core.event.base import Comment
    comments = []
    for k in ["planarity", "rectilinearity", "azimuth", "incidence"]:
        comments.append(Comment(text="%s: median = %f, mean = %f, std = %f" % (
            k, pol_stats[k]["median"], pol_stats[k]["mean"],
            pol_stats[k]["std"])))
    return comments


# Plotting

def wadati_plot(event_phases, det_st):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # Wadati plot
    for staph in event_phases:
        dep = det_st.select(station=staph["station"])[0].stats.sac["stdp"]
        sta = staph["station"]
        if staph["P"]["arrival_time"] and staph["S"]["arrival_time"]:
            tp = float(staph["P"]["arrival_time"])
            ts = float(staph["S"]["arrival_time"])
            S_P = ts - tp
            axs[0].plot(tp, S_P, marker="*", color="k")
            axs[1].plot(dep, tp, marker="*", color="k")
            axs[2].plot(dep, ts, marker="*", color="k")

            axs[0].text(tp, S_P, "%s" % sta)
            axs[1].text(dep, tp, "%s" % sta)
            axs[2].text(dep, ts, "%s" % sta)
        elif staph["S"]["arrival_time"]:
            ts = float(staph["S"]["arrival_time"])
            axs[2].plot(dep, ts, marker="*", color="k")
            axs[2].text(dep, ts, "%s" % sta)
            axs[2].text(dep, ts, "%s" % sta)

    axs[0].set_ylabel("S-P time")
    axs[0].set_xlabel("P time")
    axs[1].set_xlabel("Depth (m)")
    axs[1].set_ylabel("P time")
    axs[2].set_xlabel("Depth (m)")
    axs[2].set_ylabel("S time")
    plt.show()
    # plt.close()


def plot_phases(event_phases, det_st):
    phase_len_s = 0.2
    noise_len_s = 1.0
    tol = 0.02

    pre_pick = 1.5
    post_pick = 1.0

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for i, staph in enumerate(event_phases):
        station = staph["station"]

        if staph["P"]["arrival_time"] and staph["S"]["arrival_time"]:
            tp = staph["P"]["arrival_time"] - tol
            ts = staph["S"]["arrival_time"] - tol
            tnoise = tp - tol - 1.5 * noise_len_s
            dum = det_st.select(station=station, channel="DPZ").copy().trim(starttime=tp - pre_pick, endtime=ts - 0.02)
            # dum.detrend()
            dum.normalize()
            for tr in dum:
                axs[0].plot(tr.times("utcdatetime") - tr.stats.starttime, 0.5 * tr.data + i * 1, "k")
            axs[0].text(tr.times("utcdatetime")[-1] - tr.stats.starttime, i * 1, station)
            # Show noise window and P window
            axs[0].hlines(max(0.5 * tr.data + i * 1), tnoise - tr.stats.starttime,
                          tnoise + noise_len_s - tr.stats.starttime, color="pink", linewidth=2)
            axs[0].hlines(max(0.5 * tr.data + i * 1), tp - tr.stats.starttime, tp + phase_len_s - tr.stats.starttime,
                          color="red", linewidth=2)

        if staph["S"]["arrival_time"]:
            ts = staph["S"]["arrival_time"] - tol
            tnoise = ts - tol - 1.5 * noise_len_s
            dum = det_st.select(station=station, channel="DPZ").copy().trim(starttime=ts - pre_pick,
                                                                            endtime=ts + post_pick)
            # dum.detrend()
            dum.normalize()
            for tr in dum:
                axs[1].plot(tr.times("utcdatetime") - tr.stats.starttime, 0.5 * tr.data + i * 1, "k")
            axs[1].text(tr.times("utcdatetime")[-1] - tr.stats.starttime, i * 1, station)
            # Show noise window and S window
            axs[1].hlines(max(0.5 * tr.data + i * 1), tnoise - tr.stats.starttime,
                          tnoise + noise_len_s - tr.stats.starttime, color="pink", linewidth=2)
            axs[1].hlines(max(0.5 * tr.data + i * 1), ts - tr.stats.starttime, ts + phase_len_s - tr.stats.starttime,
                          color="purple", linewidth=2)

    plt.show()
    # plt.close()


def plot_waveform_rotation(stream, starttime, endtime, baz, inc):
    mode = "NE->RT"  # "NE->RT" or "ZNE->LQT"

    # Sort stations by increasing depth
    stations = list(set([tr.stats.station for tr in stream]))
    depths = [stream.select(station=sta)[0].stats.sac["stdp"] for sta in stations]
    stations = [sta for _, sta in sorted(zip(depths, stations))][::-1]

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 15), sharey=True, sharex=True)
    for i, sta in enumerate(stations):
        st = stream.select(station=sta).copy().trim(starttime=starttime, endtime=endtime)
        t = st[0].times("utcdatetime") - stream[0].stats.starttime
        st.normalize(global_max=True)
        #         axs[0].plot(t, st.select(channel="DPN")[0].data + i, "b")
        #         axs[1].plot(t, st.select(channel="DPE")[0].data + i, "b")
        #         axs[2].plot(t, st.select(channel="DPZ")[0].data + i, "b")

        if mode == "NE->RT":
            st.rotate("NE->RT", back_azimuth=baz, inclination=None)
            axs[0].plot(t, st.select(channel="DPR")[0].data + i, "k")
            axs[0].set_title("R")
            axs[1].plot(t, st.select(channel="DPT")[0].data + i, "k")
            axs[1].set_title("T")
            axs[2].plot(t, st.select(channel="DPZ")[0].data + i, "k")
            axs[2].set_title("Z")
        elif mode == "ZNE->LQT":
            st.rotate("ZNE->LQT", back_azimuth=baz, inclination=inc)
            axs[0].plot(t, st.select(channel="DPL")[0].data + i, "k")
            axs[0].set_title("L")
            axs[1].plot(t, st.select(channel="DPQ")[0].data + i, "k")
            axs[1].set_title("Q")
            axs[2].plot(t, st.select(channel="DPT")[0].data + i, "k")
            axs[2].set_title("T")

    axs[0].set_yticks(range(0, len(stations)))
    axs[0].set_yticklabels(stations)
    plt.show()
    # plt.close()


def plot_baz_inc(event_phases):
    fig = plt.figure(figsize=(16, 8))

    # Backazimuth plot
    ax1 = plt.subplot(121, projection='polar')
    ax1.set_theta_direction(-1)
    ax1.set_theta_zero_location("N")

    # Incidence angle plot
    ax2 = plt.subplot(122, projection='polar')
    ax2.set_theta_direction(-1)
    ax2.set_theta_zero_location("N")

    # Loop
    bazP = []
    incP = []
    bazS = []
    incS = []
    for staph in event_phases:

        if staph["P"]["arrival_time"]:
            baz = np.deg2rad(staph["P"]["pol_stats"]["azimuth"]["median"])
            bazP.append(baz)
            ax1.plot([baz, baz + np.pi], [1, 1], color="pink", linewidth=2)

            inc = np.deg2rad(staph["P"]["pol_stats"]["incidence"]["median"])
            ax2.plot([inc, inc], [0, 1], color="pink", linewidth=2)
            incP.append(inc)

        if staph["S"]["arrival_time"]:
            baz = np.deg2rad(staph["S"]["pol_stats"]["azimuth"]["median"])
            bazS.append(baz)
            ax1.plot([baz, baz + np.pi], [1, 1], color="blue", linewidth=2)

            inc = np.deg2rad(staph["S"]["pol_stats"]["incidence"]["median"])
            ax2.plot([inc, inc], [0, 1], color="blue", linewidth=2)
            incS.append(inc)

    ax1.plot([np.median(bazP), np.median(bazP) + np.pi], [1, 1], color="red", linewidth=2, linestyle="--")
    ax1.plot([np.median(bazS), np.median(bazS) + np.pi], [1, 1], color="cyan", linewidth=2, linestyle="--")
    ax1.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax1.set_xticklabels(['N', 'E', 'S', 'W'])
    ax1.set_title("Backazimuth")
    ax1.set_rticks([])

    ax2.plot([np.median(incP), np.median(incP)], [0, 1], color="red", linewidth=2, linestyle="--")
    ax2.plot([np.median(incS), np.median(incS)], [0, 1], color="cyan", linewidth=2, linestyle="--")
    ax2.set_rticks([])
    ax2.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
    ax2.set_xticklabels(['up', '', 'down', ''])
    ax2.set_title("Incidence")

    plt.show()
    # plt.close()


# Obspy & Pyrocko

def markers_to_quakeml(phase_markers):
    from obspy.core.event import Catalog, Event, Pick, CreationInfo, \
        WaveformStreamID
    from obspy.core.event.base import QuantityError, Comment
    from obspy.core.event.origin import Origin
    from obspy import UTCDateTime

    creation_string = CreationInfo(creation_time=UTCDateTime.now())

    start_time = datetime.datetime(1970, 1, 1)
    latlon_error = QuantityError(uncertainty=0.009)  # equals 1 km

    event = Event()
    event.creation_info = creation_string

    stations = []
    sta_time_tup = []
    for phase_marker in phase_markers:
        phase_station = list(phase_marker.nslc_ids)[0][1]
        stations.append(phase_station)
        phase_network = list(phase_marker.nslc_ids)[0][0]
        phase_channel = list(phase_marker.nslc_ids)[0][3]
        wav_id = WaveformStreamID(station_code=phase_station,
                                  channel_code=phase_channel,
                                  location_code="",
                                  network_code=phase_network)
        phase_name = phase_marker._phasename
        phase_tmin = phase_marker.tmin
        phase_tmax = phase_marker.tmax
        if phase_tmin == phase_tmax:
            phase_unc = 0.05
            phase_time = phase_tmin
        else:
            phase_unc = (phase_tmax - phase_tmin) * 0.5
            phase_mid_int = (phase_tmax - phase_tmin) * 0.5
            phase_time = phase_tmin + phase_mid_int

        phase_time_utc = UTCDateTime(start_time.utcfromtimestamp(phase_time))
        if phase_marker.get_polarity() == 1:
            phase_polarity = "positive"
        elif phase_marker.get_polarity() == -1:
            phase_polarity = "negative"
        else:
            phase_polarity = "undecidable"

        pick = Pick(time=phase_time_utc, phase_hint=phase_name, waveform_id=wav_id, polarity=phase_polarity,
                    time_errors=QuantityError(uncertainty=phase_unc), method_id="snuffler")
        event.picks.append(pick)
        sta_time_tup.append((wav_id.get_seed_string(), pick.time))

    # Add origin.
    # Use loc of station with earliest pick and a delay of -0.12 s, a ray travelling up from below at 250.0 m depth.
    sta_time_tup.sort(key=lambda x: x[1], reverse=False)
    earliest_sta_id = sta_time_tup[0][0]
    approx_event_time = sta_time_tup[0][1] - 0.12

    latitude = inv.get_coordinates(earliest_sta_id)["latitude"]
    longitude = inv.get_coordinates(earliest_sta_id)["longitude"]

    origin = Origin(time=approx_event_time, time_errors=QuantityError(uncertainty=0.12),
                    longitude=longitude, latitude_errors=latlon_error,
                    latitude=latitude, longitude_errors=latlon_error,
                    depth=250.0, depth_errors=QuantityError(uncertainty=250.0),
                    method_id="dummy")
    event.origins.append(origin)
    event.preferred_origin_id = str(event.origins[0].resource_id)

    cat = Catalog()
    cat.append(event)
    return cat


def picks_to_markers(picks):
    markers = []
    for p in picks:
        nscl = (p.waveform_id.network_code, p.waveform_id.station_code,
                p.waveform_id.location_code, p.waveform_id.channel_code)
        kind = 1 if p.phase_hint == "P" else 2
        pick_time = str_to_time(
            p.time.strftime("%Y-%m-%d %H:%M:") + "%f" % (p.time.second + p.time.microsecond * 1e-6))

        m = PhaseMarker(nslc_ids=[nscl], tmin=pick_time, tmax=pick_time, kind=kind, phasename=p.phase_hint)
        markers.append(m)
    return markers
