import numpy as np
from pyrolite.util.meta import subkwargs
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from ...util.log import Handle

logger = Handle(__name__)

class GeometryCollection(object):
    def __init__(self, *objects, **kwargs):
        """Container for geometry objects."""
        self.objects = []
        self.objects += list(objects)
        self.update_dict()

    def update_dict(self):
        """Generate the dictionary for referencing objects by name."""
        self._objects = {o.name: o for o in self.objects}

    @property
    def lines(self):
        return (i for i in self.objects if isinstance(i, (Linear2D, LogLinear2D)))

    @property
    def points(self):
        return (i for i in self.objects if isinstance(i, (Point)))

    def add_to_axes(self, ax, **kwargs):
        for line in self.lines:
            line.add_to_axes(ax, **kwargs)

        for p in self.points:
            p.add_to_axes(ax, **kwargs)

    def __add__(self, obj):
        """Add a component to the collection."""
        self.objects.append(obj)
        self.update_dict()
        return self

    def __getitem__(self, name):
        """Get a component referenced by name."""
        return self._objects[name]

    def __iter__(self):
        """Iterate through components."""
        return (i for i in self.objects)


class Point(object):
    def __init__(self, point, name=None, **kwargs):
        """
        Simple container for a 2D point object with basic utility functions.

        Parameters
        ----------
        point : :class:`tuple`
            x-y point.
        name : :class:`str`
            Name of the specific x-y point.

        """
        self.name = name or None
        self.x, self.y = point
        self.kwargs = kwargs

    def add_to_axes(self, ax, label=False, **kwargs):
        """
        Plot this point on an :class:`~matplotlib.axes.Axes`.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`.
            Axes to plot the line on.

        Returns
        --------
        :class:`matplotlib.collections.PathCollection`
            PathCollection as plotted on the axes.

        """
        return ax.scatter(
            self.x,
            self.y,
            label=[None, self.name][label],
            **{**self.kwargs, **subkwargs(kwargs, ax.scatter, PathCollection)}
        )


class Linear2D(object):
    def in_tfm(self, x):
        return np.array(x)

    def out_tfm(self, x):
        return np.array(x)

    def __init__(
        self,
        p0=np.array([0, 0]),
        p1=None,
        slope=None,
        name=None,
        xlim=None,
        ylim=None,
        **kwargs
    ):
        """
        Simple container for a 2D line object with basic utility functions.
        Lines can be generated from two points or a point and a slope.

        Parameters
        ----------
        p0 : :class:`numpy.ndarray` | :class:`tuple`
            An x-y point which the line passes through.
        p1 : :class:`numpy.ndarray` | :class:`tuple`
            Optionally specified x-y second point.
        slope : :class:`float`
            Optionally-specified slope of the line.
        name : :class:`str`
            Name of the specific line.
        xlim : :class:`tuple`
            Optionally-specified limits of extent in `x`.
        ylim : :class:`tuple`
            Optionally-specified limits of extent in `y`.

        Attributes
        ----------
        slope : :class:`float`
            Slope of the line.
        intercept : :class:`float`
            Y-intercept of the line.
        func
            Callable function to evaluate :math:`y = m \cdot x + c`.
        invfunc
            Callable function to evaluate :math:`x = (y - c) / m`.

        """
        self.name = name or None
        self.xlim = None
        self.ylim = None
        self.set_params(p0=p0, p1=p1, slope=slope)
        self.set_xlim(xlim)
        self.set_ylim(ylim)
        self.kwargs = kwargs

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
    def invfunc(self):
        """
        Get the function corresponding to the line parameterised as
        :math:`x = (y-c) /m`.
        """

        def line(ys):
            return (ys - self.intercept) / self.slope

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
        return self.out_tfm(np.array([x, y]))

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

        Returns
        --------
        :class:`Linear2D`
            Line instance.
        """
        return self.__class__(np.array(centre), slope=-1 / self.slope, **kwargs)

    def __call__(self, x):
        """
        Call the line function on a sequence of x values.
        """
        return self.out_tfm(self.func(self.in_tfm(x)))

    def add_to_axes(self, ax, xs=None, label=False, **kwargs):
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
            xmin, xmax = self.xlim
        elif xs is None and self.xlim is None:  # use the axes limits
            xmin, xmax = ax.get_xlim()
        else:
            xmin, xmax = np.nanmin(xs), np.nanmax(xs)

        if xs is None:
            linexs = np.logspace(np.log(xmin), np.log(xmax), 100, base=np.e)
        else:
            linexs = xs
        xmin, xmax = max(xmin, np.nanmin(linexs)), min(xmax, np.nanmax(linexs))
        ybounds = [self(xmin), self(xmax)]
        ymin, ymax = min(*ybounds), max(*ybounds)
        if self.ylim is not None:
            ymin, ymax = max(self.ylim[0], ymin), min(self.ylim[1], ymax)

        if not xmin > xmax:
            if np.abs(self.slope) > 1.0:  # more vertical than horizonal
                lineys = np.logspace(np.log(ymin), np.log(ymax), xs.size, base=np.e)
                linexs = self.out_tfm(self.invfunc(self.in_tfm(lineys)))
            else:
                lineys = self.out_tfm(self.func(self.in_tfm(linexs)))

            # fltr = np.ones(linexs.shape).astype(bool)
            # fltr = (lineys >= ymin) & (lineys <= ymax)
            # fltr = (linexs >= xmin) & (linexs <= xmax)
            # linexs, lineys = linexs[fltr], lineys[fltr]
            # append self-styling to the output, but let it be overridden
            return ax.plot(
                linexs,
                lineys,
                label=[None, self.name][label],
                **{**self.kwargs, **subkwargs(kwargs, ax.plot, Line2D)}
            )

    def __add__(self, obj):
        """Add this object to another and get a :class:`GeometryCollection`."""
        return GeometryCollection(self, obj)


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
