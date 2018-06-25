import unittest
import pandas as pd
import numpy as np
import ternary

from pyrolite.geochem import REE, common_elements
from pyrolite.plot import ternaryplot, spiderplot


class TestSpiderplot(unittest.TestCase):
    """Tests the Spiderplot functionality."""

    def setUp(self):
        reels = REE(output='string')
        self.df = pd.DataFrame({k: v for k,v in zip(reels,
                                np.random.rand(len(reels), 10))})

    def test_none(self):
        """Test generation of plot with no data."""
        pass

    def test_one(self):
        """Test generation of plot with one record."""
        pass

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        pass

    def test_no_axis_specified(self):
        """Test generation of plot without axis specified."""
        pass

    def test_axis_specified(self):
        """Test generation of plot with axis specified."""
        pass

    def test_no_components_specified(self):
        """Test generation of plot with no components specified."""
        pass

    def test_components_specified(self):
        """Test generation of plot with components specified."""
        pass

    def test_plot_off(self):
        """Test plot generation with plot off."""
        pass

    def test_fill(self):
        """Test fill functionality is available."""
        pass

    @unittest.expectedFailure
    def test_noplot_nofill(self):
        """Test failure on no-plot no-fill options."""
        spiderplot(self.df, plot=False, fill=False)

    def test_valid_style(self):
        """Test valid styling options."""
        pass

    def test_irrellevant_style_options(self):
        """Test stability under additional kwargs."""
        style = {'thingwhichisnotacolor': 'notacolor', 'irrelevant': 'red'}
        with self.assertWarns(UserWarning):
            ax = spiderplot(self.df, **style)

    @unittest.expectedFailure
    def test_invalid_style_options(self):
        """Test stability under invalid style values."""
        style = {'color': 'notacolor', 'marker': 'red'}
        spiderplot(self.df, **style)

class TestTernaryplot(unittest.TestCase):
    """Tests the Ternaryplot functionality."""

    def setUp(self):
        self.cols = ['MgO', 'CaO', 'SiO2']
        self.df = pd.DataFrame({k: v for k,v in zip(self.cols,
                                np.random.rand(len(self.cols), 10))})

    def test_none(self):
        """Test generation of plot with no data."""
        df = pd.DataFrame(columns=self.cols)
        out = ternaryplot(df)
        self.assertEqual(type(out),
                         ternary.ternary_axes_subplot.TernaryAxesSubplot)


    def test_one(self):
        """Test generation of plot with one record."""
        df = self.df.head(1)
        out = ternaryplot(df)
        self.assertEqual(type(out),
                         ternary.ternary_axes_subplot.TernaryAxesSubplot)

    def test_multiple(self):
        """Test generation of plot with multiple records."""
        df = self.df.loc[:, :]
        out = ternaryplot(df)
        self.assertEqual(type(out),
                         ternary.ternary_axes_subplot.TernaryAxesSubplot)

    def test_tax_returned(self):
        """Check that the axis item returned is a ternary axis."""
        pass

    def test_overplotting(self):
        """Test use of the plot for multiple rounds of plotting."""
        pass


if __name__ == '__main__':
    unittest.main()
