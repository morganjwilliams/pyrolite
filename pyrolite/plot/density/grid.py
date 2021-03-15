import numpy as np
from ...util.math import on_finite, linspc_, logspc_, linrng_, logrng_, flattengrid
from ...util.plot.grid import bin_centres_to_edges
from ...util.distributions import sample_kde
from ...util.log import Handle

logger = Handle(__name__)


class DensityGrid(object):
    def __init__(
        self, x, y, extent=None, bins=50, logx=False, logy=False, coverage_scale=1.2
    ):
        """
        Build a grid of x-y coordinates for use in evaluating KDE functions.

        Parameters
        -----------
        x : :class:`np.ndarray`
        y : :class:`np.ndarray`
        extent : :class:`list`
            Optionally-specified extent for the grid in the form (xmin, xmax, ymin, ymax).
        bins : :class:`int` | :class:`tuple`
            Number of bins for the grid. Can optionally specify
            a tuple with (xbins, ybins).
        logx : :class:`bool`
            Whether to use a log-spaced index for the x dimension of the grid.
        logy: :class:`bool`
            Whether to use a log-spaced index for the y dimension of the grid.
        coverage_scale : :class:`float`
            Multiplier for the range of the grid relative to the data. If >1, the grid
            will extend beyond the data.
        """
        # limits
        self.logx = logx
        self.logy = logy
        if not isinstance(bins, int):
            assert len(bins) == 2  # x-y bins
            self.xbins, self.ybins = bins
        else:
            self.xbins, self.ybins = bins, bins
        self.coverage_scale = coverage_scale

        if extent is None:
            self.xmin, self.xmax, self.ymin, self.ymax = self.extent_from_xy(x, y)
        else:
            self.xmin, self.xmax, self.ymin, self.ymax = extent

        self.xstep = self.get_xstep()
        self.ystep = self.get_ystep()

        if self.logx:
            assert self.xmin > 0.0
            assert (self.xmin / self.xstep) > 0.0
        if self.logy:
            assert self.ymin > 0.0
            assert (self.ymin / self.ystep) > 0.0

        self.calculate_grid()

    def calculate_grid(self):
        self.get_centre_grid()
        self.get_edge_grid()

    def get_ystep(self):
        if self.logy:
            return (self.ymax / self.ymin) / self.ybins
        else:
            return (self.ymax - self.ymin) / self.ybins

    def get_xstep(self):
        if self.logx:
            return (self.xmax / self.xmin) / self.xbins
        else:
            return (self.xmax - self.xmin) / self.xbins

    def extent_from_xy(self, x, y, coverage_scale=None):
        cov = coverage_scale or self.coverage_scale
        expand_grid = (cov - 1.0) / 2
        return [
            *[linrng_, logrng_][self.logx](x, exp=expand_grid),
            *[linrng_, logrng_][self.logy](y, exp=expand_grid),
        ]

    def get_xrange(self):
        return self.xmin, self.xmax

    def get_yrange(self):
        return self.ymin, self.ymax

    def get_extent(self):
        return [*self.get_xrange(), *self.get_yrange()]

    def get_range(self):
        return [[*self.get_xrange()], [*self.get_yrange()]]

    def update_grid_centre_ticks(self):
        if self.logx:
            self.grid_xc = logspc_(self.xmin, self.xmax, 1.0, self.xbins)
        else:
            self.grid_xc = linspc_(self.xmin, self.xmax, self.xstep, self.xbins)

        if self.logy:
            self.grid_yc = logspc_(self.ymin, self.ymax, 1.0, self.ybins)
        else:
            self.grid_yc = linspc_(self.ymin, self.ymax, self.ystep, self.ybins)

    def update_grid_edge_ticks(self):
        self.update_grid_centre_ticks()
        if self.logx:
            self.grid_xe = np.exp(bin_centres_to_edges(np.log(np.sort(self.grid_xc))))
        else:
            self.grid_xe = bin_centres_to_edges(np.sort(self.grid_xc))

        if self.logy:
            self.grid_ye = np.exp(bin_centres_to_edges(np.log(np.sort(self.grid_yc))))
        else:
            self.grid_ye = bin_centres_to_edges(np.sort(self.grid_yc))

    def get_centre_grid(self):
        self.update_grid_centre_ticks()
        self.grid_xci, self.grid_yci = np.meshgrid(self.grid_xc, self.grid_yc)

    def get_edge_grid(self):
        self.update_grid_edge_ticks()
        self.grid_xei, self.grid_yei = np.meshgrid(self.grid_xe, self.grid_ye)

    def get_hex_extent(self):
        if self.logx:
            xex = [np.log(self.xmin / self.xstep), np.log(self.xmax * self.xstep)]
        else:
            xex = [self.xmin - self.xstep, self.xmax + self.xstep]

        if self.logy:
            yex = [np.log(self.ymin / self.ystep), np.log(self.ymax * self.ystep)]
        else:
            yex = [self.ymin - self.ystep, self.ymax + self.ystep]
        return xex + yex

    def kdefrom(
        self,
        xy,
        xtransform=lambda x: x,
        ytransform=lambda x: x,
        mode="centres",
        bw_method=None,
    ):
        """
        Take an x-y array and sample a KDE on the grid.
        """
        arr = xy.copy()
        # generate x grid over range spanned by log(x)
        arr[:, 0] = xtransform(xy[:, 0])
        # generate y grid over range spanned by log(y)
        arr[:, 1] = ytransform(xy[:, 1])
        if mode == "centres":
            assert np.isfinite(self.grid_xc).all() and np.isfinite(self.grid_yc).all()
            zi = sample_kde(
                arr,
                flattengrid(
                    np.meshgrid(xtransform(self.grid_xc), ytransform(self.grid_yc))
                ),
                bw_method=bw_method,
            )
            zi = zi.reshape(self.grid_xci.shape)
        elif mode == "edges":
            assert np.isfinite(self.grid_xe).all() and np.isfinite(self.grid_ye).all()
            zi = sample_kde(
                arr,
                flattengrid(
                    np.meshgrid(xtransform(self.grid_xe), ytransform(self.grid_ye))
                ),
                bw_method=bw_method,
            )
            zi = zi.reshape(self.grid_xei.shape)
        else:
            raise NotImplementedError("Valid modes are 'centres' and 'edges'.")
        return zi
