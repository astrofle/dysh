from collections import UserList
from copy import deepcopy

import astropy.units as u
import numpy as np
from astropy.wcs import WCS

from ..util import sq_weighted_avg
from . import average, mean_tsys, tsys_weight, veldef_to_convention
from .spectrum import Spectrum


class ScanMixin(object):
    @property
    def status(self):
        """Status flag, will be used later for undo"""
        return self._status

    @property
    def nrows(self):
        """The number of rows in this Scan"""
        return self._nrows

    @property
    def npol(self):
        """The number of polarizations in this Scan"""
        return self._npol

    def calibrate(self, **kwargs):
        pass

    def timeaverage(self, weights=None):
        pass

    def polaverage(self, weights=None):
        pass

    def finalspectum(self, weights=None):
        pass

    def __len__(self):
        return self._nrows


class ScanBlock(UserList, ScanMixin):
    def __init__(self, *args):
        super().__init__(*args)
        self._status = None
        self._nrows = 0
        self._npol = 0
        self._timeaveraged = []
        self._polaveraged = []
        self._finalspectrum = []

    def calibrate(self, **kwargs):
        for scan in self.data:
            scan.calibrate(**kwargs)

    def timeaverage(self, weights="tsys"):
        for scan in self.data:
            self._timeaveraged.append(scan.timeaverage(weights))
        return self._timeaveraged

    def polaverage(self, weights="tsys"):
        for scan in self.data:
            self._polaveraged.append(scan.polaverage(weights))
        return self._polaveraged

    def finalspectum(self, weights="tsys"):
        for scan in self.data:
            self._finalspectrum.append(scan.finalspectrum(weights))
        return self._finalspectrum


class TPScan(ScanMixin):
    """GBT specific version of Total Power Scan

    Parameters
    ----------
    gbtfits : `~fits.gbtfitsload.GBFITSLoad`
        input GBFITSLoad object
    scan: int
        scan number
    sigstate : str
        one of 'SIG' or 'REF' to indicate if this is the signal or reference scan or 'BOTH' if it contains both
    calstate : str
        one of 'ON' or 'OFF' to indicate the calibration state of this scan, or 'BOTH' if it contains both
    scanrows : list-like
        the list of rows in `sdfits` corresponding to sig_state integrations
    calrows : dict
        dictionary containing with keys 'ON' and 'OFF' containing list of rows in `sdfits` corresponding to cal=T (ON) and cal=F (OFF) integrations for `scan`
    bintable : int
        the index for BINTABLE in `sdfits` containing the scans
    calibrate: bool
        whether or not to calibrate the data.  If `True`, the data will be (calon - caloff)*0.5, otherwise it will be SDFITS row data. Default:True
    """

    # @TODO get rid of calrows and calc tsys in gettp and pass it in.
    def __init__(self, gbtfits, scan, sigstate, calstate, scanrows, calrows, bintable, calibrate=True):
        self._sdfits = gbtfits  # parent class
        self._scan = scan
        self._sigstate = sigstate  # ignored?
        self._calstate = calstate  # ignored?
        self._scanrows = scanrows
        print("BINTABLE = ", bintable)
        # @TODO deal with data that crosses bintables
        if bintable is None:
            self._bintable_index = gbtfits._find_bintable_and_row(self._scanrows[0])[0]
        else:
            self._bintable_index = bintable
        self._data = self._sdfits.rawspectra(self._bintable_index)[scanrows]  # all cal states
        self._status = 0  # @TODO make these an enumeration, possibly dict
        #                           # ex1:
        self._nint = 0
        self._npol = 0
        self._timeaveraged = None
        self._polaveraged = None
        # self._nrows = len(scanrows)
        self._tsys = None
        if False:
            self._npol = gbtfits.npol(self._bintable_index)  # TODO deal with bintable
            self._nint = gbtfits.nintegrations(self._bintable_index)
        self._calrows = calrows
        self._refonrows = self._calrows["ON"]
        self._refoffrows = self._calrows["OFF"]
        self._refcalon = gbtfits.rawspectra(self._bintable_index)[self._refonrows]
        self._refcaloff = gbtfits.rawspectra(self._bintable_index)[self._refoffrows]
        self._calibrate = calibrate
        if self._calibrate:
            self._data = 0.5 * (self._refcalon + self._refcaloff)
        # print(f"# scanrows {len(self._scanrows)}, # calrows ON {len(self._calrows['ON'])}  # calrows OFF {len(self._calrows['OFF'])}")
        self.calc_tsys()

    @property
    def sigstate(self):
        return self._sigstate

    @property
    def calstate(self):
        return self._calstate

    @property
    def tsys(self):
        """The system temperature array.

        Returns
        -------
        tsys : `~numpy.ndarray`
            System temperature values in K
        """
        return self._tsys

    def calc_tsys(self, **kwargs):
        """
        Calculate the system temperature array
        """
        kwargs_opts = {"verbose": False}
        kwargs_opts.update(kwargs)

        self._status = 1

        tcal = list(self._sdfits.index(bintable=self._bintable_index).iloc[self._refonrows]["TCAL"])
        nspect = len(tcal)
        self._tsys = np.empty(nspect, dtype=float)  # should be same as len(calon)
        # allcal = self._refonrows.copy()
        # allcal.extend(self._refoffrows)
        # tcal = list(self._sdfits.index(self._bintable_index).iloc[sorted(allcal)]["TCAL"])
        # @Todo  this loop could be replaces with clever numpy
        if len(tcal) != nspect:
            raise Exception(f"TCAL length {len(tcal)} and number of spectra {nspect} don't match")
        for i in range(nspect):
            tsys = mean_tsys(calon=self._refcalon[i], caloff=self._refcaloff[i], tcal=tcal[i])
            self._tsys[i] = tsys

    @property
    def exposure(self):
        """Get the array of exposure (integration) times

            exposure =  0.5*(exp_ref_on + exp_ref_off)

        Note we only have access to the refon and refoff row indices so can't use sig here.
        This is probably incorrect

        Returns
        -------
            exposure : `~numpy.ndarray`
                The exposure time in units of the EXPOSURE keyword in the SDFITS header
        """
        exp_ref_on = self._sdfits.index(bintable=self._bintable_index).iloc[self._refonrows]["EXPOSURE"].to_numpy()
        exp_ref_off = self._sdfits.index(bintable=self._bintable_index).iloc[self._refoffrows]["EXPOSURE"].to_numpy()
        exposure = exp_ref_on + exp_ref_off
        return exposure

    @property
    def delta_freq(self):
        """Get the array of channel frequency width

           df =  0.5*(df_ref_on + df_ref_off)

        Note we only have access to the refon and refoff row indices so can't use sig here.
        This is probably incorrect

        Returns
        -------
            delta_freq: `~numpy.ndarray`
                The channel frequency width in units of the CDELT1 keyword in the SDFITS header
        """
        df_ref_on = self._sdfits.index(bintable=self._bintable_index).iloc[self._refonrows]["CDELT1"].to_numpy()
        df_ref_off = self._sdfits.index(bintable=self._bintable_index).iloc[self._refoffrows]["CDELT1"].to_numpy()
        delta_freq = 0.5 * (df_ref_on + df_ref_off)
        return delta_freq

    @property
    def _tsys_weight(self):
        r"""The system temperature weighting array computed from current
        :math:`T_{sys}, t_{exp}`, and `\delta\nu`. See :meth:`tsys_weight`
        """
        return tsys_weight(self.exposure, self.delta_freq, self.tsys)

    def total_power(self, i):
        """Return the total power spectrum

        Parameters
        ----------
            i : int
                The index into the data array

        Returns
        -------
            spectrum : `~spectra.spectrum.Spectrum`
        """
        meta = dict(self._sdfits.index(bintable=self._bintable_index).iloc[self._scanrows[i]])
        meta["TSYS"] = self._tsys[i]
        meta["EXPOSURE"] = self.exposure[i]
        naxis1 = len(self._data[i])
        ctype1 = meta["CTYPE1"]
        ctype2 = meta["CTYPE2"]
        ctype3 = meta["CTYPE3"]
        crval1 = meta["CRVAL1"]
        crval2 = meta["CRVAL2"]
        crval3 = meta["CRVAL3"]
        crpix1 = meta["CRPIX1"]
        cdelt1 = meta["CDELT1"]
        restfrq = meta["RESTFREQ"]
        if "CUNIT1" in meta:
            cunit1 = meta["CUNIT1"]
        else:
            cunit1 = "Hz"  # @TODO this is in gbtfits.hdu[0].header['TUNIT11'] but is it always TUNIT11?
        rfq = restfrq * u.Unit(cunit1)
        restfreq = rfq.to("Hz").value

        # @TODO WCS is expensive.  Figure how to calculate spectral_axis instead.
        wcs = WCS(
            header={
                "CDELT1": cdelt1,
                "CRVAL1": crval1,
                "CUNIT1": cunit1,
                "CTYPE1": "FREQ",
                "CRPIX1": crpix1,
                "RESTFRQ": restfreq,
                "CTYPE2": ctype2,
                "CRVAL2": crval2,
                "CRPIX2": 1,
                "CTYPE3": ctype3,
                "CRVAL3": crval3,
                "CRPIX3": 1,
                "CUNIT2": "deg",
                "CUNIT3": "deg",
                "NAXIS1": naxis1,
                "NAXIS2": 1,
                "NAXIS3": 1,
            },
        )
        vc = veldef_to_convention(meta["VELDEF"])
        s = Spectrum(self._data[i] * u.ct, wcs=wcs, meta=meta, velocity_convention=vc)
        return s

    def timeaverage(self, weights="tsys"):
        r"""Compute the time-averaged spectrum for this set of scans.

        Parameters
        ----------
                weights: str
                    'tsys' or None.  If 'tsys' the weight will be calculated as:

                     :math:`w = t_{exp} \times \delta\nu/T_{sys}^2`

                    Default: 'tsys'
        Returns
        -------
                spectrum : :class:`~spectra.spectrum.Spectrum`
                    The time-averaged spectrum
        """
        self._timeaveraged = deepcopy(self.total_power(0))
        if weights == "tsys":
            w = self._tsys_weight
        else:
            w = None
        self._timeaveraged._data = average(self._data, axis=0, weights=w)
        self._timeaveraged.meta["MEANTSYS"] = np.mean(self._tsys)
        self._timeaveraged.meta["WTTSYS"] = sq_weighted_avg(self._tsys, axis=0, weights=w)
        self._timeaveraged.meta["TSYS"] = self._timeaveraged.meta["WTTSYS"]
        self._timeaveraged.meta["EXPOSURE"] = self.exposure.sum()
        return self._timeaveraged


class PSScan(ScanMixin):
    """GBT specific version of Position Switch Scan

    Parameters
    ----------

    gbtfits : `~fit.gbtfitsload.GBFITSLoad`
        input GBFITSLoad object
    scans : dict
        dictionary with keys 'ON' and 'OFF' containing unique list of ON (signal) and OFF (reference) scan numbers
    scanrows : dict
        dictionary with keys 'ON' and 'OFF' containing the list of rows in `sdfits` corresponding to ON (signal) and OFF (reference) integrations
    calrows : dict
        dictionary containing with keys 'ON' and 'OFF' containing list of rows in `sdfits` corresponding to cal=T (ON) and cal=F (OFF) integrations.
    bintable : int
        the index for BINTABLE in `sdfits` containing the scans
    """

    def __init__(self, gbtfits, scans, scanrows, calrows, bintable):
        # PSScan.__init__(self, gbtfits, scans, scanrows, bintable)
        # The rows of the original bintable corresponding to ON (sig) and OFF (reg)
        self._sdfits = gbtfits  # parent class
        self._scans = scans
        self._scanrows = scanrows
        self._nrows = len(self._scanrows["ON"])
        # print(f"scanrows ON {self._scanrows['ON']}")
        # print(f"scanrows OFF {self._scanrows['OFF']}")

        # calrows perhaps not needed as input since we can get it from gbtfits object?
        # calrows['ON'] are rows with noise diode was on, regardless of sig or ref
        # calrows['OFF'] are rows with noise diode was off, regardless of sig or ref
        self._calrows = calrows
        print("BINTABLE = ", bintable)
        # @TODO deal with data that crosses bintables
        if bintable is None:
            self._bintable_index = gbtfits._find_bintable_and_row(self._scanrows["ON"][0])[0]
        else:
            self._bintable_index = bintable
        if False:
            self._npol = gbtfits.npol(self._bintable_index)  # TODO deal with bintable
            self._nint = gbtfits.nintegrations(self._bintable_index)
        # todo use gbtfits.velocity_convention(veldef,velframe)
        vc = "doppler_radio"
        # so quick with slicing!
        self._sigonrows = sorted(list(set(self._calrows["ON"]).intersection(set(self._scanrows["ON"]))))
        self._sigoffrows = sorted(list(set(self._calrows["OFF"]).intersection(set(self._scanrows["ON"]))))
        self._refonrows = sorted(list(set(self._calrows["ON"]).intersection(set(self._scanrows["OFF"]))))
        self._refoffrows = sorted(list(set(self._calrows["OFF"]).intersection(set(self._scanrows["OFF"]))))
        self._sigcalon = gbtfits.rawspectra(self._bintable_index)[self._sigonrows]
        self._sigcaloff = gbtfits.rawspectra(self._bintable_index)[self._sigoffrows]
        self._refcalon = gbtfits.rawspectra(self._bintable_index)[self._refonrows]
        self._refcaloff = gbtfits.rawspectra(self._bintable_index)[self._refoffrows]
        self._tsys = None

    @property
    def tsys(self):
        """The system temperature array. This will be `None` until calibration is done.

        Returns
        -------
        tsys : `~numpy.ndarray`
            System temperature values in K
        """
        return self._tsys

    # TODO something clever
    # self._calibrated_spectrum = Spectrum(self._calibrated,...) [assuming same spectral axis]
    def calibrated(self, i):
        """Return the calibrated Spectrum.

        Parameters
        ----------
            i : int
                The index into the calibrated array

        Returns
        -------
            spectrum : `~spectra.spectrum.Spectrum`
        """
        meta = dict(self._sdfits.index(bintable=self._bintable_index).iloc[self._scanrows["ON"][i]])
        meta["TSYS"] = self._tsys[i]
        naxis1 = len(self._calibrated[i])
        ctype1 = meta["CTYPE1"]
        ctype2 = meta["CTYPE2"]
        ctype3 = meta["CTYPE3"]
        crval1 = meta["CRVAL1"]
        crval2 = meta["CRVAL2"]
        crval3 = meta["CRVAL3"]
        crpix1 = meta["CRPIX1"]
        cdelt1 = meta["CDELT1"]
        restfrq = meta["RESTFREQ"]
        if "CUNIT1" in meta:
            cunit1 = meta["CUNIT1"]
        else:
            cunit1 = "Hz"  # @TODO this is in gbtfits.hdu[0].header['TUNIT11'] but is it always TUNIT11?
        rfq = restfrq * u.Unit(cunit1)
        restfreq = rfq.to("Hz").value

        # @TODO WCS is expensive.  Figure how to calculate spectral_axis instead.
        wcs = WCS(
            header={
                "CDELT1": cdelt1,
                "CRVAL1": crval1,
                "CUNIT1": cunit1,
                "CTYPE1": "FREQ",
                "CRPIX1": crpix1,
                "RESTFRQ": restfreq,
                "CTYPE2": ctype2,
                "CRVAL2": crval2,
                "CRPIX2": 1,
                "CTYPE3": ctype3,
                "CRVAL3": crval3,
                "CRPIX3": 1,
                "CUNIT2": "deg",
                "CUNIT3": "deg",
                "NAXIS1": naxis1,
                "NAXIS2": 1,
                "NAXIS3": 1,
            },
        )
        vc = veldef_to_convention(meta["VELDEF"])

        return Spectrum(self._calibrated[i] * u.K, wcs=wcs, meta=meta, velocity_convention=vc)

    def calibrate(self, **kwargs):
        """
        Position switch calibration, following equations 1 and 2 in the GBTIDL clibration manual
        """
        kwargs_opts = {"verbose": False}
        kwargs_opts.update(kwargs)

        self._status = 1
        nspect = self.nrows // 2
        self._calibrated = np.empty(nspect, dtype=np.ndarray)
        self._tsys = np.empty(nspect, dtype=float)

        tcal = list(self._sdfits.index(bintable=self._bintable_index).iloc[self._refonrows]["TCAL"])
        # @Todo  this loop could be replaced with clever numpy
        if len(tcal) != nspect:
            raise Exception(f"TCAL length {len(tcal)} and number of spectra {nspect} don't match")
        for i in range(nspect):
            tsys = mean_tsys(calon=self._refcalon[i], caloff=self._refcaloff[i], tcal=tcal[i])
            sig = 0.5 * (self._sigcalon[i] + self._sigcaloff[i])
            ref = 0.5 * (self._refcalon[i] + self._refcaloff[i])
            self._calibrated[i] = tsys * (sig - ref) / ref
            self._tsys[i] = tsys

    # tip o' the hat to Pedro S. for exposure and delta_freq
    @property
    def exposure(self):
        """Get the array of exposure (integration) times

        exposure = [ 0.5*(exp_ref_on + exp_ref_off) + 0.5*(exp_sig_on + exp_sig_off) ] / 2

        Returns
        -------
             exposure : ~numpy.ndarray
                 The exposure time in units of the EXPOSURE keyword in the SDFITS header
        """
        exp_ref_on = self._sdfits.index(bintable=self._bintable_index).iloc[self._refonrows]["EXPOSURE"].to_numpy()
        exp_ref_off = self._sdfits.index(bintable=self._bintable_index).iloc[self._refoffrows]["EXPOSURE"].to_numpy()
        exp_sig_on = self._sdfits.index(bintable=self._bintable_index).iloc[self._sigonrows]["EXPOSURE"].to_numpy()
        exp_sig_off = self._sdfits.index(bintable=self._bintable_index).iloc[self._sigoffrows]["EXPOSURE"].to_numpy()
        exp_ref = exp_ref_on + exp_ref_off
        exp_sig = exp_sig_on + exp_sig_off
        # exposure = 0.5*(exp_ref + exp_sig)
        # exposure = exp_ref + exp_sig
        nsmooth = 1.0  # In case we start smoothing the reference spectra.
        exposure = exp_sig * exp_ref * nsmooth / (exp_sig + exp_ref * nsmooth)
        return exposure

    @property
    def delta_freq(self):
        """Get the array of channel frequency width

        df = [ 0.5*(df_ref_on + df_ref_off) + 0.5*(df_sig_on + df_sig_off) ] / 2

        Returns
        -------
             delta_freq: ~numpy.ndarray
                 The channel frequency width in units of the CDELT1 keyword in the SDFITS header
        """
        df_ref_on = self._sdfits.index(bintable=self._bintable_index).iloc[self._refonrows]["CDELT1"].to_numpy()
        df_ref_off = self._sdfits.index(bintable=self._bintable_index).iloc[self._refoffrows]["CDELT1"].to_numpy()
        df_sig_on = self._sdfits.index(bintable=self._bintable_index).iloc[self._sigonrows]["CDELT1"].to_numpy()
        df_sig_off = self._sdfits.index(bintable=self._bintable_index).iloc[self._sigoffrows]["CDELT1"].to_numpy()
        df_ref = 0.5 * (df_ref_on + df_ref_off)
        df_sig = 0.5 * (df_sig_on + df_sig_off)
        delta_freq = 0.5 * (df_ref + df_sig)
        return delta_freq

    @property
    def _tsys_weight(self):
        r"""The system temperature weighting array computed from current
        :math`T_{sys}`, :math:`t_{int}`, and :math:`\delta\nu`. See :meth:`tsys_weight`
        """
        return tsys_weight(self.exposure, self.delta_freq, self.tsys)

    def timeaverage(self, weights="tsys"):
        r"""Compute the time-averaged spectrum for this set of scans.

        Parameters
        ----------
                weights: str
                    'tsys' or None.  If 'tsys' the weight will be calculated as:

                     :math:`w = t_{exp} \times \delta\nu/T_{sys}^2`

                    Default: 'tsys'
        Returns
        -------
                spectrum : :class:`~spectra.spectrum.Spectrum`
                    The time-averaged spectrum
        """
        if self._calibrated is None:
            raise Exception("You can't time average before calibration.")
        self._timeaveraged = deepcopy(self.calibrated(0))
        data = self._calibrated
        if weights == "tsys":
            w = self._tsys_weight
        else:
            w = None
        self._timeaveraged._data = average(data, axis=0, weights=w)
        self._timeaveraged.meta["MEANTSYS"] = np.mean(self._tsys)
        self._timeaveraged.meta["WTTSYS"] = sq_weighted_avg(self._tsys, axis=0, weights=w)
        self._timeaveraged.meta["TSYS"] = self._timeaveraged.meta["WTTSYS"]
        return self._timeaveraged
