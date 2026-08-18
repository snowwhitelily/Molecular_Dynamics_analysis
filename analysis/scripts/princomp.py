# princomp.py -- PCA/plotting utilities by T. A. Wassenaar
import numpy as np
from pymol import cmd
from colorinator import *

# Boxplot with marker for median
#
#                      |--------[============||============]--------|
BOXPLOT = (          0.00,    0.25,    0.48,    0.52,    0.75,    1.00)
BOXRADI = (               0.10,    0.30,    0.20,    0.30,    0.10   )
FIVECOL = np.array(((0,1,1), (0,1,0),     (1,1,0),     (1,0,0), (1,0,1)))


# NOTE on CGO objects.
# Units are given as
# - cylinder(14): [  9.0, x,y,z, x,y,z, R, r,g,b, r,g,b]
# - cone(17):     [ 27.0, x,y,z, x,y,z, R, R, r,g,b, r,g,b, c, c]
# - sphere(9):    [  6.0, r,g,b, 7.0, x,y,z, R ]
CGOSIZE = {'cylinders': 14, 'cones': 17, 'spheres': 9}
CGOCOORD = {'cylinders': [1, 4], 'cones': [1, 4], 'spheres': 5}
CGOCOLOR = {'cylinders': [8, 11], 'cones': [9, 12], 'spheres': 1}
CGOSIZE = {'cylinders': 7, 'cones': [7, 8], 'spheres': 8}


## E.g., violin.recolor(BWR, pc1.scores, P.states).draw('pc1violin')

class AtomicCGO:
    def __init__(self, atoms=None, points=None, **kwargs):

        # The atoms correspond to the first axis of the CGO contents array
        # If this is None, only index-based selections are possible.
        self.atoms = atoms

        # The points are the quantile values that correspond to the cylinder 
        # ends. If they are None, the points are assumed to be the (non)linear
        # quantile points.
        self.points = points

        for kw in ('spheres', 'cylinders', 'cones'):
            setattr(self, kw, kwargs.pop(kw, None))
            
        # Organization of each of these will be natoms, points, unit
        for kw, val in kwargs.items():
            setattr(self, kw, val)
                
    def __len__(self):
        return len(self.points)

    def __getitem__(self, item):
        if isinstance(item, str):
            return AtomicCGO(self.base, item=getattr(self, item))
        # Otherwise make a selection in all the items

    def __imul__(self, fac):
        if self.cylinders is not None:
            self.cylinders[..., 7] *= fac
        if self.spheres is not None:
            self.spheres[..., 4] *= fac
        if self.cones is not None:
            self.cones[..., 7] *= fac
            self.cones[..., 8] *= fac        
        return self

    def __imod__(self, col):
        if isinstance(col, (float, int)) or (len(col) == 3 and isinstance(col[0], (float, int))):
            col = (col, col)
            
        if self.cylinders is not None:
            self.cylinders[..., 8:11] = col[:-1]
            self.cylinders[..., 11:14] = col[1:]
        if self.cones is not None:
            self.cones[..., 9:12] = col[:-1]
            self.cones[..., 12:15] = col[1:]
        if self.spheres is not None:
            ...
            
        return self
            
    def draw(self, name='atomiccgo', scale=None, color=None):

        if scale is not None:
            self *= scale

        if color is not None:
            self %= color
            
        cgo = []
        for attr in ('spheres', 'cylinders', 'cones'):
            elements = getattr(self, attr)
            if elements is not None:
                cgo.extend(elements.flatten())
        cmd.load_cgo(cgo, name)

    def recolor(self, colors=BWR.kde, x=None, y=None, bw=10):
        """Recolor cylinders by average or KDE-weighted property values.

        Parameters
        ----------
            colors : float | tuple | array | Colorinator
                Color mapping object.
            y : array-like
                Property value for each frame (e.g., frame indices, time, RMSD).
            bw : float, default=0.1
                Bandwidth for KDE weighting (independent of violin density bw).

        """

        clen = hasattr(colors, '__len__') and len(colors)
        
        # Simplest case first: matching colors directly
        # Single value (gray) color or single rgb color tuple or matching points
        # This does catch Colorinators of length equal to number of points
        if (
                isinstance(colors, (int, float)) or
                (clen == 3 and isinstance(colors[0], (int, float))) or
                clen == len(self.points)
        ):
            self %= colors
            return self
        
        # If there's a list, tuple or array of colors make a Colorinator.
        # If x is provided, they are assumed to be quantile points
        # Note that a Colorinator instance is also callable
        if not callable(colors):
            colors = Colorinator(colors, x)

        # If the thing is simply a Colorinator, the colors of it
        # are to be matched with the available points. Cylinders 
        # and such will need to be split.
        # Colors are mapped along self.points, which must share the same
        # parameterization as the Colorinator.points.
        if isinstance(colors, Colorinator):
            newcolors = np.vstack([colors(self.points), colors.points])
            allpoints = np.concatenate((self.points, colors.points))
            ndx = allpoints.argsort()
            ...
            raise NotImplementedError
            return ...

        # If we end up here we have a function that should provide
        # a coloring for given points based on the relation between
        # the points and another feature. The binding is based on
        # values x, which should be in the same space as the points
        # now in the CGO object.
        self %= colors(x, y, self.points)
        return self

    
## E.g., pc1 = P[1]; violin = pc1.violin()
    
class CartesianComponent:
    """Represents a single principal component with geometric utilities.
    
    Stores mean structure, loading vector, and projection scores for one
    principal component, with methods to generate PyMOL CGO visualizations.
    
    Parameters
    ----------
    positions : ndarray of shape (n_atoms, 3)
        Mean atomic coordinates.
    loadings : ndarray of shape (n_atoms * 3,)
        Principal component loading vector (flattened coordinates).
    scores : ndarray of shape (n_frames,)
        Projection of each frame onto this component.
    model : chempy.models.Indexed, optional
        PyMOL model for atom metadata.
    """
    def __init__(self, positions, loadings, scores, model=None):
        self.positions = positions
        self.loadings = loadings
        self.scores = scores
        self.model = model
    
    def cgo(self, points, radii, colors=None):
        """Generate CGO representation along the component trajectory.
        
        Creates cylinder segments between consecutive points along the
        principal component motion, with per-segment radii and colors.
        
        Parameters
        ----------
        points : array-like of shape (n_points,)
            Positions along the component to evaluate (in score units).
        radii : float or array-like of shape (n_points - 1,)
            Radius for each cylinder segment. If scalar, applied uniformly.
        colors : ndarray of shape (n_points, 3), optional
            RGB colors in [0, 1] for each point. Segments interpolate
            between consecutive point colors.
        
        Returns
        -------
        AtomicCGO
            CGO object containing cylinders for each atom's motion.
        
        Raises
        ------
        ValueError
            If number of radii doesn't match number of segments.
        """
        if isinstance(radii, (int, float)):
            radii = [ radii ] * (len(points) - 1)
        if len(radii) != len(points) - 1:
            raise ValueError(f'Incompatible number of radii ({len(radii)}) for number of segments ({len(points) - 1})')
        Q = self.positions + points[:, None, None] * self.loadings.reshape(1, -1, 3)
        # CGO object array - atoms, points, cylinders
        C = np.ones((len(self.positions), len(points)-1, 14))
        C[..., 1:4] = Q[:-1].transpose((1, 0, 2))
        C[..., 4:7] = Q[1:].transpose((1, 0, 2))
        C[..., 0] = 9.0
        C[..., 7] = radii
        if colors is not None:
            C[..., 8:11] = colors[:-1]
            C[..., 11:14] = colors[1:]
        return AtomicCGO(self.model, points, values=self.scores, cylinders=C)
    
    def box(self, radii=BOXRADI, colors=BOX):
        """Generate boxplot-style visualization of score distribution.
        
        Creates cylinders at quartile positions (min, Q1, median, Q3, max)
        with specified radii and colors.
        
        Parameters
        ----------
        radii : array-like of length 4, default=BOXRADI
            Radii for the four segments (min-Q1, Q1-median, median-Q3, Q3-max).
        colors : ndarray of shape (5, 3), default=FIVECOL
            RGB colors for the five quartile points.
        
        Returns
        -------
        AtomicCGO
            CGO representation of the boxplot.
        """
        return self.cgo(np.quantile(self.scores, BOXPLOT), radii, colors)
    
    def violin(self, bins=50, bw=10, radius=0.3, colors=BOX):
        """Generate violin plot visualization of score distribution.
        
        Creates a density-modulated representation using kernel density
        estimation, where cylinder radius reflects local score density.
        
        Parameters
        ----------
        bins : int, default=50
            Number of evaluation points for density estimation.
        bw : float, default=10
            Bandwidth for Gaussian kernel density estimation.
        radius : float, default=0.3
            Maximum radius (scaled by density).
        colors : ndarray of shape (5, 3), default=FIVECOL
            RGB colors for the five quartile points.
        
        Returns
        -------
        AtomicCGO
            CGO representation of the violin plot.
        """
        x = np.linspace(self.scores.min() - 1e-6, self.scores.max() + 1e-6, 2*bins + 1)
        d = np.exp(((self.scores[:, None] - x[None, 1::2]) ** 2) / (- bw**2)).sum(axis=0)
        return self.cgo(x[::2], radius * d / d.max(), BOX.map(x[::2]))

    def minmax(self, radius=0.05, colors=0.3):
        if isinstance(colors, (int, float)):
            colors = [colors, colors]
        return self.cgo(np.quantile(self.scores, (0, 1)), radius, colors)
    

class CoordinateArray:
    def __init__(self,  selection='not hydro', frames=slice(None)):
        # Collection of coordinates
        self.selection = selection
        objects = cmd.get_object_list(selection)
        self.model = cmd.get_model(f'{objects[0]} and ({selection})')
        states = [ np.arange(cmd.count_states(obj))[frames] for obj in objects ]
        objects = [ np.repeat(obj, len(sts)) for obj, sts in zip(objects, states) ]
        self.objects = np.array([ frame for obj in objects for frame in obj ])
        self.states = np.array([ s for sts in states for s in sts ])
        self._X = None

    @property
    def X(self, cache=False):
        # Extraction of coordinates. Maybe the above (and maybe the following)
        # should be a separate object.
        if self._X is not None:
            return self._X
        
        X = np.array([ 
            cmd.get_model(f'{obj} and {self.selection}', state=s).get_coord_list()
            for obj, s in zip(self.objects, self.states)
        ])

        if cache:
            self._X = X

        return X

    
## e.g., P = Princomp('not hydro')

class Princomp:
    """Perform PCA on molecular structures in PyMOL.
    
    Analyzes structural variance across frames by decomposing atomic coordinates
    into principal components. Structures should be pre-aligned.
    
    Parameters
    ----------
    selection : str, default='not hydro'
        PyMOL atom selection to include in the analysis.
    frames : slice, default=slice(None)
        Slice object specifying which frames to include from each object.
    ncomponents : int, default=10
        Number of principal components to compute scores for.
    
    Attributes
    ----------
    selection : str
        Original selection
    model : chempy.models.Indexed
        PyMOL model of the selection from the first object.
    objects : ndarray of str
        Object names for each analyzed frame.
    states : ndarray of int
        State numbers for each analyzed frame.
    mean : ndarray of shape (n_atoms, 3)
        Mean coordinates across all frames.
    variances : ndarray
        Eigenvalues (variance explained) for all components, descending order.
    loadings : ndarray of shape (n_atoms * 3, n_components)
        Principal component vectors (eigenvectors).
    scores : ndarray of shape (n_frames, n_components)
        Projection of each frame onto the principal components.
    """
    def __init__(self, selection='not hydro', frames=slice(None), ncomponents=10):

        self.data = CoordinateArray(selection, frames)
        X = self.data.X

        # Collection of coordinates
        #self.selection = selection
        #objects = cmd.get_object_list(selection)
        #self.model = cmd.get_model(f'{objects[0]} and ({selection})')
        #states = [ np.arange(cmd.count_states(obj))[frames] for obj in objects ]
        #objects = [ np.repeat(obj, len(sts)) for obj, sts in zip(objects, states) ]
        #self.objects = np.array([ frame for obj in objects for frame in obj ])
        #self.states = np.array([ s for sts in states for s in sts ])
        # Extraction of coordinates. Maybe the above (and maybe the following)
        # should be a separate object.
        #X = np.array([ 
        #    cmd.get_model(f'{obj} and {selection}', state=s).get_coord_list()
        #    for obj, s in zip(self.objects, self.states)
        #])
        
        # PCA
        self.mean = X.mean(axis=0)
        self.var = X.var(axis=0)
        Z = (X - self.mean).reshape((len(X), -1))
        
        if len(Z) < self.mean.size:
            D, P = np.linalg.eigh(Z @ (Z.T / len(Z)))
            P = Z.T @  P
            P /= (P ** 2).sum(axis=0) ** 0.5
        else:
            D, P = np.linalg.eigh(Z.T @ (Z / len(Z)))
        self.variances = D[::-1]
        self.loadings = P[:, ::-1]
        self.scores = Z @ P[:, -1:-1-ncomponents:-1] 
    
    def __len__(self):
        """Return the number of principal components with computed scores."""
        return self.scores.shape[1]
            
    def __getitem__(self, item):
        """Access a principal component by index.
        
        Parameters
        ----------
        item : int
            Component number (1-indexed, e.g., 1 for PC1).
        
        Returns
        -------
        CartesianComponent
            Object containing mean structure, loading vector, and scores
            for the requested component.
        
        Raises
        ------
        TypeError
            If item is not an integer.
        IndexError
            If component index is out of valid range [1, n_components].
        """
        if not isinstance(item, int):
            raise TypeError('Expected integer component index.')
        # This should also be able to take a string selection over atoms
        # and return a Princomp object of that selection
        
        ncomp = len(self)
        if item < 1:
            raise IndexError(f'Component indices start at 1. Maximum component is {ncomp}.')
        if item > ncomp:
            raise IndexError(f'Component index exceeds number of components for which scores are available ({ncomp})')
        compidx = item - 1
        return CartesianComponent(self.mean, self.loadings[:, compidx], self.scores[:, compidx])

    def drawmean(self, name='pcmean'):
        for i,j,k in zip(self.data.model.atom, self.mean, self.var.sum(axis=1)):
            i.coord[:] = j
            i.b = k
        cmd.load_model(self.data.model, name)

    def project(selection, frames=None, components=slice(None)):
        ...

    def filter(selection=None, frames=None, components=slice(None)):
        ...
        
