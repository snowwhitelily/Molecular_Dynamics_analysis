# colorinator.py -- colour utilities by T. A. Wassenaar (princomp/PyMOL toolkit)
import numpy as np
#from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
#from skimage.color import rgb2lab, lab2rgb


# Boxplot with marker for median
#
#                      |--------[============||============]--------|
BOXPLOT = (          0.00,    0.25,    0.48,    0.52,    0.75,    1.00)
BOXRADI = (               0.10,    0.30,    0.20,    0.30,    0.10   )
FIVECOL = np.array(((0,1,1), (0,1,0),     (1,1,0),     (1,0,0), (1,0,1)))


def mix_mean(colors):
    return np.nanmean(colors, axis=0)

def mix_geom(colors):
    return np.exp(np.nanmean(np.log(np.clip(colors, 1e-12, 1)), axis=0))

def mix_rms(colors):
    return np.sqrt(np.nanmean(colors**2, axis=0))

def mix_median(colors):
    return np.nanmedian(colors, axis=0)

#def mix_hsv(colors):
#    hsv = rgb_to_hsv(colors)
#    mean = np.nanmean(hsv, axis=0)
#    return hsv_to_rgb(mean[None, :])[0]

#def mix_lab(colors):
#    lab = rgb2lab(colors[None, :, :])[0]
#    mean = np.nanmean(lab, axis=0)
#    return lab2rgb(mean[None, None, :])[0, 0]

MIX_METHODS = {
    'mean': mix_mean,      # regular average
    'geom': mix_geom,      # geometric mean
    'rms': mix_rms,        # emphasizes bright colors
    'median': mix_median,  # robust against outliers
#    'hsv': mix_hsv,        # hue–saturation–value averaging
#    'lab': mix_lab,        # perceptually uniform averaging
}


class Colorinator:
    """Maps data values to colors with support for non-uniform color scales.
    
    Useful for creating color gradients where averaging produces meaningful
    intermediate colors (e.g., blue-white-red for negative-neutral-positive).

    points define the control points of the color scale. These are not
    restricted to quantiles — they can represent any monotonic
    parameterization (e.g., distances, component magnitudes, or
    normalized coordinates).
    
    Parameters
    ----------
    colors : array-like of shape (n_colors, 3) or (n_colors, 4)
        RGB or RGBA color values in [0, 1] range.
    points : array-like of shape (n_colors,), optional
        Control positions of colors. If None, colors are evenly spaced
        from 0 to 1.
    """
    def __init__(self, colors, points=None):
        q = np.arange(len(colors)) / (len(colors) - 1)
        if points is None:
            points = q
        if len(colors) != len(points):
            colors = np.array([ np.interp(points, q, c) for c in colors.T ]).T
        self.colors = np.array(colors)
        self.points = points

    def __call__(self, points, pmin=0, pmax=1):
        return self.map(points, pmin, pmax)
        
    def __len__(self):
        return len(self.colors)

    def __getitem__(self, item):
        return self.colors[item]
    
    def map(self, points, pmin=None, pmax=None):
        """Map data values to colors.
        
        Values are normalized to [0, 1] based on the range of the input,
        then interpolated against the color gradient.
        
        Parameters
        ----------
        points : array-like
            Data values to map to colors.
        
        Returns
        -------
        ndarray of shape (n_points, n_channels)
            Color values for each input point.
        """
        points = np.array(points)
        if pmin is None:
            pmin = points.min()
        if pmax is None:
            pmax = points.max()
        points = (points - pmin) / (pmax - pmin)
        return np.array([ np.interp(points, self.points, c) for c in self.colors.T ]).T
    
    def average(self, x, y, points=50):
        """Compute average colors in bins along x.
        
        Parameters
        ----------
        x : array-like
            Positions for binning.
        y : array-like
            Data values to map to colors and average.
        points : int or array-like, default=50
            If int, number of equally-spaced bins. If array, bin edges.
        
        Returns
        -------
        ndarray of shape (n_bins, n_channels)
            Average color in each bin. Empty bins appear black.
        """
        colors = self.map(y)
        if isinstance(points, int):
            npoints = points + 1
            bins = (points * (x - x.min() - 1e-10) / (x.max() - x.min() + 1e-10)).astype(int)
        else:
            npoints = len(points) + 1
            bins = np.digitize(x, points)
        counts = np.bincount(bins, minlength=npoints)[:-1]
        counts[counts == 0] = 1
        colsum = np.array([ np.bincount(bins, weights=c, minlength=npoints)[:-1] for c in colors.T ])
        return (colsum / counts).T

    def _average(self, x, y, points=50, mix='mean'):
        colors = self.map(y)
        mixfn = MIX_METHODS.get(mix, mix_mean)

        if isinstance(points, int):
            npoints = points + 1
            bins = (points * (x - x.min() - 1e-10) / (x.max() - x.min() + 1e-10)).astype(int)
        else:
            npoints = len(points) + 1
            bins = np.digitize(x, points)

        results = []
        for i in range(npoints - 1):
            sel = bins == i
            if not np.any(sel):
                results.append(np.zeros(3))
            else:
                results.append(mixfn(colors[sel]))
        return np.array(results)

    def kde(self, x, y, points=50, bw=10):
        """Compute kernel density estimate of colors along x.
        
        Uses Gaussian kernels to smooth color values spatially.
        
        Parameters
        ----------
        x : array-like
            Positions of data points.
        y : array-like
            Data values to map to colors.
        points : int or array-like, default=50
            If int, number of equally-spaced evaluation points. If array,
            explicit evaluation positions.
        bw : float, default=10
            Bandwidth (standard deviation) of Gaussian kernel.
        
        Returns
        -------
        ndarray of shape (n_points, n_channels)
            Smoothed color values at evaluation points.
        """
        if isinstance(points, int):
            points = np.linspace(x.min(), x.max(), points)
        weights = np.exp(((points[:, None] - x)**2) / (- bw**2))
        sums = weights.sum(axis=1)
        sums[sums == 0] = 1
        return (weights @ self.map(y)) / sums[:, None]

    def _kde(self, x, y, points=50, bw=10, mix='mean'):
        if isinstance(points, int):
            points = np.linspace(x.min(), x.max(), points)
        weights = np.exp(((points[:, None] - x)**2) / (- bw**2))
        colors = self.map(y)
        mixfn = MIX_METHODS.get(mix, mix_mean)

        results = []
        for i, w in enumerate(weights):
            if w.sum() == 0:
                results.append(np.zeros(3))
            else:
                weighted = colors * w[:, None]
                results.append(mixfn(weighted / (w.sum() + 1e-12)))
        return np.array(results)

    
BWR = Colorinator([
    (0.0, 0.0, 1.0),   # blue
    (1.0, 1.0, 1.0),   # white
    (1.0, 0.0, 0.0)    # red
])

PRGn = Colorinator([
    (0.45, 0.00, 0.55),  # purple
    (1.00, 1.00, 1.00),  # white
    (0.00, 0.55, 0.35)   # green
])

BGW = Colorinator([
    (0.00, 0.20, 0.90),  # blue
    (0.20, 0.80, 0.60),  # turquoise
    (1.00, 1.00, 1.00)   # white
])

RYW = Colorinator([
    (0.80, 0.00, 0.00),  # red
    (1.00, 0.80, 0.00),  # yellow
    (1.00, 1.00, 1.00)   # white
])

PMG = Colorinator([
    (0.30, 0.00, 0.60),  # purple
    (0.90, 0.00, 0.70),  # magenta
    (0.00, 0.70, 0.30)   # green
])

Spectral = Colorinator([
    (0.00, 0.00, 0.90),  # blue
    (0.00, 0.70, 1.00),  # cyan
    (0.00, 1.00, 0.30),  # green
    (1.00, 1.00, 0.00),  # yellow
    (1.00, 0.00, 0.00)   # red
])

HWC = Colorinator([
    (1.00, 0.00, 0.50),  # hotpink
    (1.00, 1.00, 1.00),  # white
    (0.50, 1.00, 0.00)   # chartreuse
])

BOX = Colorinator(colors=FIVECOL, points=BOXPLOT)


