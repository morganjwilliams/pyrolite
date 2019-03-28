import numpy as np


class GeometryCollection(object):
    """
    Container for geometry objects.
    """

    def __init__(self, *objects, **kwargs):
        self.objects = []
        self.objects += list(objects)
        self.update_dict()

    def update_dict(self):
        self._objects = {o.name: o for o in self.objects}

    @property
    def lines(self):
        return (i for i in self.objects if isinstance(i, (Linear2D, LogLinear2D)))

    def add_to_axes(self, ax, **kwargs):
        for line in self.lines:
            line.add_to_axes(ax, **kwargs)

    def __add__(self, object):
        self.objects.append(object)
        self.update_dict()
        return self

    def __getitem__(self, name):
        return self._objects[name]

    def __iter__(self):
        return (i for i in self.objects)


class Linear2D(object):
    """
    Simple container for a 2D line object with basic utility functions.

    Attributes
    ----------
    intercept
    func
    """

    def in_tfm(self, x):
        return np.array(x)

    def out_tfm(self, x):
        return np.array(x)

    def __init__(
        self, p0=np.array([0, 0]), p1=None, slope=None, name=None, xlim=None, ylim=None
    ):
        # todo: add ability to define loglinear lines
        self.name = name or None
        self.xlim = None
        self.ylim = None
        self.set_params(p0=p0, p1=p1, slope=slope)
        self.set_xlim(xlim)
        self.set_ylim(ylim)

    def set_params(self, p0=np.array([0, 0]), p1=None, slope=None):
        self.p0 = self.in_tfm(np.array(p0))
        assert not ((p1 is None) and (slope is None))
        if p1 is not None:
            self.p1 = self.in_tfm(np.array(p1))
            diff = self.p1 - self.p0
            self.slope = diff[1] / diff[0]
        else:
            self.slope = slope
            self.p1 = None

    @property
    def intercept(self):
        """Intercept of the line on the y axis."""
        return self.p0[1] - self.slope * self.p0[0]

    @property
    def func(self):
        """Get the function corresponding to the line."""

        def line(xs):
            return xs * self.slope + self.intercept

        return line

    @property
    def equation(self):
        return " y = {slope} x + {c}".format(slope=self.slope, c=self.intercept)

    def invert_axes(self):
        """Reflect the line through the plane x==y."""
        self.p0 = self.p0[::-1]
        self.slope = 1.0 / self.slope
        self.y0 = self.intercept
        if self.p1 is not None:
            self.p1 = self.p1[::-1]
        if self.xlim is not None:
            self.ylim = self.xlim
        if self.ylim is not None:
            self.xlim = self.ylim

    def intersect(self, line):
        x = (line.intercept - self.intercept) / (self.slope - line.slope)
        y = self.func(x)
        return np.array([x, y])

    def set_xlim(self, xlim):
        # get the x value of the line intersection if a line is passed
        if xlim is not None:
            points = [
                self.intersect(x)[0] if isinstance(x, Linear2D) else x for x in xlim
            ]
            self.xlim = np.min(points), np.max(points)

    def set_ylim(self, ylim):
        # get the x value of the line intersection if a line is passed
        if ylim is not None:
            points = [
                self.intersect(y)[1] if isinstance(y, Linear2D) else y for y in ylim
            ]
            self.ylim = np.min(points), np.max(points)

    def perpendicular_line(self, centre, **kwargs):
        """
        Get a line perpendicular to this one which passes through a specified centre.

        Parameters
        -----------
        centre : :class:`numpy.ndarray`
            Array containing the point which the perpendicular line passes through.
        """
        return self.__class__(np.array(centre), slope=-1 / self.slope, **kwargs)

    def __call__(self, x):
        """
        Call the line function on a sequence of x values.
        """
        return self.out_tfm(self.func(self.in_tfm(x)))

    def add_to_axes(self, ax, xs=None, **kwargs):
        """
        Plot this line on an :class:`~matplotlib.axes.Axes`.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`.
            Axes to plot the line on.
        xs : :class:`numpy.ndarray`
            X values at which to evaluate the line function.

        Returns
        --------
        :class:`matplotlib.lines.Line2D`
            Lines as plotted on the axes.

        Todo
        -----
            * Update to draw lines based along points along their length

        Notes
        ------
            * If no x values are specified, the function will attempt to use the
                validity limits of the line, or finally use the limits of the axes.
        """
        if xs is None and self.xlim is not None:
            xlim = self.xlim
            xs = np.logspace(*np.log(xlim), 100, base=np.e)
        elif xs is None and self.xlim is None:  # use the axes limits
            xlim = ax.get_xlim()
            xs = np.logspace(*np.log(xlim), 100, base=np.e)
        else:
            xlim = np.nanmin(xs), np.nanmax(xs)

        # range validation
        xmin, xmax = max(xlim[0], np.nanmin(xs)), min(xlim[1], np.nanmax(xs))
        if not xmin > xmax:
            linexs = np.logspace(xmin, xmax, xs.size, base=np.e)
            ys = self.out_tfm(self.func(self.in_tfm(xs)))
            if self.ylim is not None:
                fltr = np.ones(xs.shape).astype(bool)
                fltr = (ys >= self.ylim[0]) & (ys <= self.ylim[1])
                xs, ys = xs[fltr], ys[fltr]
            return ax.plot(xs, ys, **kwargs)

    def __add__(self, object):
        return GeometryCollection(self, object)


class LogLinear2D(Linear2D):
    """
    Simple container for a 2D line object with basic utility functions.

    Attributes
    ----------
    intercept
    func
    """

    def in_tfm(self, x):
        return np.log(x)

    def out_tfm(self, x):
        return np.exp(x)
