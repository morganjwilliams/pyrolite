import unittest
import matplotlib.lines

from pyrolite.util.plot.style import (
    marker_cycle,
    linekwargs,
    scatterkwargs,
    patchkwargs,
    _mpl_sp_kw_split,
)


class TestMarkerCycle(unittest.TestCase):
    def test_iterable(self):
        mkrs = marker_cycle()
        for i in range(15):
            mkr = next(mkrs)

    def test_makes_line(self):
        mkrs = marker_cycle()
        for i in range(10):
            matplotlib.lines.Line2D([0], [0], marker=next(mkrs))


if __name__ == "__main__":
    unittest.main()
