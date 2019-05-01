import unittest
from pyrolite.util.distributions import lognorm_to_norm, norm_to_lognorm


class TestLognorm2Norm(unittest.TestCase):
    def setUp(self):
        self.mu = 4.0
        self.s = 1.0

    def test_default(self):
        mean, sigma = lognorm_to_norm(self.mu, self.s)
        self.assertTrue(mean >= self.mu)
        self.assertTrue(sigma > self.s)


class TestNorm2Lognorm(unittest.TestCase):
    def setUp(self):
        self.mean = 4.0
        self.sigma = 1.0

    def test_default(self):
        mu, s = norm_to_lognorm(self.mean, self.sigma)
        self.assertTrue(mu <= self.mean)
        self.assertTrue(s <= self.sigma)

    def test_scipy(self):
        for exp in [True, False]:
            with self.subTest(exp=exp):
                mu, s = norm_to_lognorm(self.mean, self.sigma, exp=exp)
                self.assertTrue(mu <= self.mean)
                self.assertTrue(s <= self.sigma)


if __name__ == "__main__":
    unittest.main()
