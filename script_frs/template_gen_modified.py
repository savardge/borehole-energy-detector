import numpy as np
import logging
import os

from obspy import Stream, read, Trace, UTCDateTime, read_events
from obspy.core.event import Catalog
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.seishub import Client as SeisHubClient

from eqcorrscan.core.template_gen import _rms, _group_events, _download_from_client
from eqcorrscan.utils.sac_util import sactoevent
from eqcorrscan.utils import pre_processing
from eqcorrscan.core import EQcorrscanDeprecationWarning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
Logger = logging.getLogger(__name__)


def custom_template_gen(method, lowcut, highcut, samp_rate, filt_order,
                        length, prepick, swin="all", process_len=86400,
                        all_horiz=False, delayed=True, plot=False, plotdir=None,
                        return_event=False, min_snr=None, parallel=False,
                        num_cores=False, save_progress=False, skip_short_chans=False,
                        **kwargs):
    """
    Generate processed and cut waveforms for use as templates.

    :type method: str
    :param method:
        Template generation method, must be one of ('from_client',
        'from_seishub', 'from_sac', 'from_meta_file'). - Each method requires
        associated arguments, see note below.
    :type lowcut: float
    :param lowcut: Low cut (Hz), if set to None will not apply a lowcut.
    :type highcut: float
    :param highcut: High cut (Hz), if set to None will not apply a highcut.
    :type samp_rate: float
    :param samp_rate: New sampling rate in Hz.
    :type filt_order: int
    :param filt_order: Filter level (number of corners).
    :type length: float
    :param length: Length of template waveform in seconds.
    :type prepick: float
    :param prepick: Pre-pick time in seconds
    :type swin: str
    :param swin:
        P, S, P_all, S_all or all, defaults to all: see note in
        :func:`eqcorrscan.core.template_gen.template_gen`
    :type process_len: int
    :param process_len: Length of data in seconds to download and process.
    :type all_horiz: bool
    :param all_horiz:
        To use both horizontal channels even if there is only a pick on one of
        them.  Defaults to False.
    :type delayed: bool
    :param delayed: If True, each channel will begin relative to it's own \
        pick-time, if set to False, each channel will begin at the same time.
    :type plot: bool
    :param plot: Plot templates or not.
    :type plotdir: str
    :param plotdir:
        The path to save plots to. If `plotdir=None` (default) then the figure
        will be shown on screen.
    :type return_event: bool
    :param return_event: Whether to return the event and process length or not.
    :type min_snr: float
    :param min_snr:
        Minimum signal-to-noise ratio for a channel to be included in the
        template, where signal-to-noise ratio is calculated as the ratio of
        the maximum amplitude in the template window to the rms amplitude in
        the whole window given.
    :type parallel: bool
    :param parallel: Whether to process data in parallel or not.
    :type num_cores: int
    :param num_cores:
        Number of cores to try and use, if False and parallel=True, will use
        either all your cores, or as many traces as in the data (whichever is
        smaller).
    :type save_progress: bool
    :param save_progress:
        Whether to save the resulting templates at every data step or not.
        Useful for long-running processes.
    :type skip_short_chans: bool
    :param skip_short_chans:
        Whether to ignore channels that have insufficient length data or not.
        Useful when the quality of data is not known, e.g. when downloading
        old, possibly triggered data from a datacentre

    :returns: List of :class:`obspy.core.stream.Stream` Templates
    :rtype: list

    """

    client_map = {'from_client': 'fdsn', 'from_seishub': 'seishub'}
    assert method in ('from_client', 'from_seishub', 'from_meta_file',
                      'from_sac')
    if not isinstance(swin, list):
        swin = [swin]
    process = True
    if method in ['from_client', 'from_seishub']:
        catalog = kwargs.get('catalog', Catalog())
        data_pad = kwargs.get('data_pad', 90)
        # Group catalog into days and only download the data once per day
        sub_catalogs = _group_events(
            catalog=catalog, process_len=process_len, template_length=length,
            data_pad=data_pad)
        if method == 'from_client':
            if isinstance(kwargs.get('client_id'), str):
                client = FDSNClient(kwargs.get('client_id', None))
            else:
                client = kwargs.get('client_id', None)
            available_stations = []
        else:
            client = SeisHubClient(kwargs.get('url', None), timeout=10)
            available_stations = client.waveform.get_station_ids()
    elif method == 'from_meta_file':
        if isinstance(kwargs.get('meta_file'), Catalog):
            catalog = kwargs.get('meta_file')
        elif kwargs.get('meta_file'):
            catalog = read_events(kwargs.get('meta_file'))
        else:
            catalog = kwargs.get('catalog')
        sub_catalogs = [catalog]
        st = kwargs.get('st', Stream())
        process = kwargs.get('process', True)
    elif method == 'from_sac':
        sac_files = kwargs.get('sac_files')
        if isinstance(sac_files, list):
            if isinstance(sac_files[0], (Stream, Trace)):
                # This is a list of streams...
                st = Stream(sac_files[0])
                for sac_file in sac_files[1:]:
                    st += sac_file
            else:
                sac_files = [read(sac_file)[0] for sac_file in sac_files]
                st = Stream(sac_files)
        else:
            st = sac_files
        # Make an event object...
        catalog = Catalog([sactoevent(st)])
        sub_catalogs = [catalog]

    temp_list = []
    process_lengths = []
    catalog_out = Catalog()

    if "P_all" in swin or "S_all" in swin or all_horiz:
        all_channels = True
    else:
        all_channels = False
    for sub_catalog in sub_catalogs:
        if method in ['from_seishub', 'from_client']:
            Logger.info("Downloading data")
            st = _download_from_client(
                client=client, client_type=client_map[method],
                catalog=sub_catalog, data_pad=data_pad,
                process_len=process_len, available_stations=available_stations,
                all_channels=all_channels)
        Logger.info('Pre-processing data')
        st.merge()
        if len(st) == 0:
            Logger.info("No data")
            continue
        if process:
            data_len = max([len(tr.data) / tr.stats.sampling_rate
                            for tr in st])
            if 80000 < data_len < 90000:
                daylong = True
                starttime = min([tr.stats.starttime for tr in st])
                min_delta = min([tr.stats.delta for tr in st])
                # Cope with the common starttime less than 1 sample before the
                #  start of day.
                if (starttime + min_delta).date > starttime.date:
                    starttime = (starttime + min_delta)
                # Check if this is stupid:
                if abs(starttime - UTCDateTime(starttime.date)) > 600:
                    daylong = False
                starttime = starttime.date
            else:
                daylong = False
            # Check if the required amount of data have been downloaded - skip
            # channels if arg set.
            for tr in st:
                if np.ma.is_masked(tr.data):
                    _len = np.ma.count(tr.data) * tr.stats.delta
                else:
                    _len = tr.stats.npts * tr.stats.delta
                if _len < process_len * .8:
                    Logger.info(
                        "Data for {0} are too short, skipping".format(
                            tr.id))
                    if skip_short_chans:
                        continue
                # Trim to enforce process-len
                tr.data = tr.data[0:int(process_len * tr.stats.sampling_rate)]
            if len(st) == 0:
                Logger.info("No data")
                continue
            if daylong:
                st = pre_processing.dayproc(
                    st=st, lowcut=lowcut, highcut=highcut,
                    filt_order=filt_order, samp_rate=samp_rate,
                    parallel=parallel, starttime=UTCDateTime(starttime),
                    num_cores=num_cores)
            else:
                st = pre_processing.shortproc(
                    st=st, lowcut=lowcut, highcut=highcut,
                    filt_order=filt_order, parallel=parallel,
                    samp_rate=samp_rate, num_cores=num_cores)
        data_start = min([tr.stats.starttime for tr in st])
        data_end = max([tr.stats.endtime for tr in st])

        for event in sub_catalog:
            stations, channels, st_stachans = ([], [], [])
            if len(event.picks) == 0:
                Logger.warning(
                    'No picks for event {0}'.format(event.resource_id))
                continue
            use_event = True
            # Check that the event is within the data
            for pick in event.picks:
                if not data_start < pick.time < data_end:
                    Logger.warning(
                        "Pick outside of data span: Pick time {0} Start "
                        "time {1} End time: {2}".format(
                            str(pick.time), str(data_start), str(data_end)))
                    use_event = False
            if not use_event:
                Logger.error('Event is not within data time-span')
                continue
            # Read in pick info
            Logger.debug("I have found the following picks")
            for pick in event.picks:
                if not pick.waveform_id:
                    Logger.warning(
                        'Pick not associated with waveforms, will not use:'
                        ' {0}'.format(pick))
                    continue
                Logger.debug(pick)
                stations.append(pick.waveform_id.station_code)
                channels.append(pick.waveform_id.channel_code)
            # Check to see if all picks have a corresponding waveform
            for tr in st:
                st_stachans.append('.'.join([tr.stats.station,
                                             tr.stats.channel]))
            # Cut and extract the templates
            template = _template_gen(
                event.picks, st, length, swin, prepick=prepick, plot=plot,
                all_horiz=all_horiz, delayed=delayed, min_snr=min_snr,
                plotdir=plotdir)
            process_lengths.append(len(st[0].data) / samp_rate)
            temp_list.append(template)
            catalog_out += event
        if save_progress:
            if not os.path.isdir("eqcorrscan_temporary_templates"):
                os.makedirs("eqcorrscan_temporary_templates")
            for template in temp_list:
                template.write(
                    "eqcorrscan_temporary_templates{0}{1}.ms".format(
                        os.path.sep, template[0].stats.starttime.strftime(
                            "%Y-%m-%dT%H%M%S")),
                    format="MSEED")
        del st
    if return_event:
        return temp_list, catalog_out, process_lengths
    return temp_list


def _template_gen(picks, st, length, swin='all', prepick=0.05,
                  all_horiz=False, delayed=True, plot=False, min_snr=None,
                  plotdir=None):
    """
    Master function to generate a multiplexed template for a single event.

    Function to generate a cut template as :class:`obspy.core.stream.Stream`
    from a given set of picks and data.  Should be given pre-processed
    data (downsampled and filtered).

    :type picks: list
    :param picks: Picks to extract data around, where each pick in the \
        list is an obspy.core.event.origin.Pick object.
    :type st: obspy.core.stream.Stream
    :param st: Stream to extract templates from
    :type length: float
    :param length: Length of template in seconds
    :type swin: str
    :param swin:
        P, S, P_all, S_all or all, defaults to all: see note in
        :func:`eqcorrscan.core.template_gen.template_gen`
    :type prepick: float
    :param prepick:
        Length in seconds to extract before the pick time default is 0.05
        seconds.
    :type all_horiz: bool
    :param all_horiz:
        To use both horizontal channels even if there is only a pick on one
        of them.  Defaults to False.
    :type delayed: bool
    :param delayed:
        If True, each channel will begin relative to it's own pick-time, if
        set to False, each channel will begin at the same time.
    :type plot: bool
    :param plot:
        To plot the template or not, default is False. Plots are saved as
        `template-starttime_template.png` and `template-starttime_noise.png`,
        where `template-starttime` is the start-time of the template
    :type min_snr: float
    :param min_snr:
        Minimum signal-to-noise ratio for a channel to be included in the
        template, where signal-to-noise ratio is calculated as the ratio of
        the maximum amplitude in the template window to the rms amplitude in
        the whole window given.
    :type plotdir: str
    :param plotdir:
        The path to save plots to. If `plotdir=None` (default) then the figure
        will be shown on screen.

    :returns: Newly cut template.
    :rtype: :class:`obspy.core.stream.Stream`

    .. note::
        By convention templates are generated with P-phases on the
        vertical channel and S-phases on the horizontal channels, normal
        seismograph naming conventions are assumed, where Z denotes vertical
        and N, E, R, T, 1 and 2 denote horizontal channels, either oriented
        or not.  To this end we will **only** use Z channels if they have a
        P-pick, and will use one or other horizontal channels **only** if
        there is an S-pick on it.

    .. note::
        swin argument: Setting to `P` will return only data for channels
        with P picks, starting at the pick time (minus the prepick).
        Setting to `S` will return only data for channels with
        S picks, starting at the S-pick time (minus the prepick)
        (except if `all_horiz=True` when all horizontal channels will
        be returned if there is an S pick on one of them). Setting to `all`
        will return channels with either a P or S pick (including both
        horizontals if `all_horiz=True`) - with this option vertical channels
        will start at the P-pick (minus the prepick) and horizontal channels
        will start at the S-pick time (minus the prepick).
        `P_all` will return cut traces starting at the P-pick time for all
        channels. `S_all` will return cut traces starting at the S-pick
        time for all channels.

    .. warning::
        If there is no phase_hint included in picks, and swin=all, all
        channels with picks will be used.
    """
    from eqcorrscan.utils.plotting import pretty_template_plot as tplot
    from eqcorrscan.utils.plotting import noise_plot

    # the users picks intact.
    if not isinstance(swin, list):
        swin = [swin]
    for _swin in swin:
        assert _swin in ['P', 'all', 'S', 'P_all', 'S_all']
    picks_copy = []
    for pick in picks:
        if not pick.waveform_id:
            Logger.warning(
                "Pick not associated with waveform, will not use it: "
                "{0}".format(pick))
            continue
        if not pick.waveform_id.station_code or not \
                pick.waveform_id.channel_code:
            Logger.warning(
                "Pick not associated with a channel, will not use it:"
                " {0}".format(pick))
            continue
        picks_copy.append(pick)
    if len(picks_copy) == 0:
        return Stream()
    st_copy = Stream()
    for tr in st:
        # Check that the data can be represented by float16, and check they
        # are not all zeros
        if np.all(tr.data.astype(np.float16) == 0):
            Logger.error("Trace is all zeros at float16 level, either gain or "
                         "check. Not using in template: {0}".format(tr))
            continue
        st_copy += tr
    st = st_copy
    if len(st) == 0:
        return st
    # Get the earliest pick-time and use that if we are not using delayed.
    picks_copy.sort(key=lambda p: p.time)
    first_pick = picks_copy[0]
    if plot:
        stplot = st.slice(first_pick.time - 2,
                          first_pick.time + length + 2).copy()
        noise = stplot.copy()
    # Work out starttimes
    starttimes = []
    for _swin in swin:
        for tr in st:
            starttime = {'station': tr.stats.station,
                         'channel': tr.stats.channel, 'picks': []}
            station_picks = [pick for pick in picks_copy
                             if pick.waveform_id.station_code ==
                             tr.stats.station]
            if _swin == 'P_all':
                p_pick = [pick for pick in station_picks
                          if pick.phase_hint.upper()[0] == 'P']
                if len(p_pick) == 0:
                    continue
                starttime.update({'picks': p_pick})
            elif _swin == 'S_all':
                s_pick = [pick for pick in station_picks
                          if pick.phase_hint.upper()[0] == 'S']
                if len(s_pick) == 0:
                    continue
                starttime.update({'picks': s_pick})
            elif _swin == 'all':
                if all_horiz and tr.stats.channel[-1] in ['1', '2', '3',
                                                          'N', 'E']:
                    # Get all picks on horizontal channels
                    channel_pick = [
                        pick for pick in station_picks
                        if pick.waveform_id.channel_code[-1] in
                           ['1', '2', '3', 'N', 'E']]
                else:
                    channel_pick = [
                        pick for pick in station_picks
                        if pick.waveform_id.channel_code == tr.stats.channel]
                if len(channel_pick) == 0:
                    continue
                starttime.update({'picks': channel_pick})
            elif _swin == 'P':
                p_pick = [pick for pick in station_picks
                          if pick.phase_hint.upper()[0] == 'P' and
                          pick.waveform_id.channel_code == tr.stats.channel]
                if len(p_pick) == 0:
                    continue
                starttime.update({'picks': p_pick})
            elif _swin == 'S':
                s_pick = [pick for pick in station_picks
                          if pick.phase_hint.upper()[0] == 'S']
                if not all_horiz:
                    s_pick = [pick for pick in s_pick
                              if pick.waveform_id.channel_code ==
                              tr.stats.channel]
                starttime.update({'picks': s_pick})
                if len(starttime['picks']) == 0:
                    continue
            if not delayed:
                starttime.update({'picks': [first_pick]})
            starttimes.append(starttime)
    # Cut the data
    st1 = Stream()
    for _starttime in starttimes:
        Logger.info(f"Working on channel {_starttime['station']}."
                    f"{_starttime['channel']}")
        tr = st.select(
            station=_starttime['station'], channel=_starttime['channel'])[0]
        Logger.info(f"Found Trace {tr}")
        used_tr = False
        for pick in _starttime['picks']:
            if not pick.phase_hint:
                Logger.warning(
                    "Pick for {0}.{1} has no phase hint given, you should not "
                    "use this template for cross-correlation"
                    " re-picking!".format(
                        pick.waveform_id.station_code,
                        pick.waveform_id.channel_code))
            starttime = pick.time - prepick
            Logger.debug("Cutting {0}".format(tr.id))
            noise_amp = _rms(
                tr.slice(starttime=starttime - 100, endtime=starttime).data)
            tr_cut = tr.slice(
                starttime=starttime, endtime=starttime + length,
                nearest_sample=False).copy()
            if plot:
                noise.select(
                    station=_starttime['station'],
                    channel=_starttime['channel']).trim(
                    noise[0].stats.starttime, starttime)
            if len(tr_cut.data) == 0:
                Logger.warning(
                    "No data provided for {0}.{1} starting at {2}".format(
                        tr.stats.station, tr.stats.channel, starttime))
                continue
            # Ensure that the template is the correct length
            if len(tr_cut.data) == (tr_cut.stats.sampling_rate *
                                    length) + 1:
                tr_cut.data = tr_cut.data[0:-1]
            Logger.debug(
                'Cut starttime = %s\nCut endtime %s' %
                (str(tr_cut.stats.starttime), str(tr_cut.stats.endtime)))
            if min_snr is not None and \
                    max(tr_cut.data) / noise_amp < min_snr:
                Logger.warning(
                    "Signal-to-noise ratio {0} below threshold for {1}.{2}, "
                    "not using".format(
                        max(tr_cut.data) / noise_amp, tr_cut.stats.station,
                        tr_cut.stats.channel))
                continue
            st1 += tr_cut
            used_tr = True
        if not used_tr:
            Logger.warning('No pick for {0}'.format(tr.id))
    if plot and len(st1) > 0:
        plot_kwargs = dict(show=True)
        if plotdir is not None:
            if not os.path.isdir(plotdir):
                os.makedirs(plotdir)
            plot_kwargs.update(dict(show=False, save=True))
        tplot(st1, background=stplot, picks=picks_copy,
              title='Template for ' + str(st1[0].stats.starttime),
              savefile="{0}/{1}_template.png".format(
                  plotdir, st1[0].stats.starttime.strftime(
                      "%Y-%m-%dT%H%M%S")),
              **plot_kwargs)
        noise_plot(signal=st1, noise=noise,
                   savefile="{0}/{1}_noise.png".format(
                       plotdir, st1[0].stats.starttime.strftime(
                           "%Y-%m-%dT%H%M%S")),
                   **plot_kwargs)
        del stplot
    return st1
