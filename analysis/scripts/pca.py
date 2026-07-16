import time
import ast
from pymol import cmd

"""
PCA module for Pymol
"""

__author__  = "Tsjerk A. Wassenaar"
__version__ = "1.0"


import numpy


## GENERAL STUFF ##

_R2SMALL = 1e-16

def unpack(lst): 
    """Unpack a nested list"""
    return [ i for j in lst for i in (unpack(j) if hasattr(j,"__iter__") else [j]) ]


def normalized(v):
    """Normalize vector"""
    return v/numpy.linalg.norm(v)


def parse_keyword_function(s, *args):
    """Generate a function from a string with arguments"""
    basefun = "def tmpfun(%s):\n    return %s"
    loc = {}
    print(help(exec))
    exec(basefun % (','.join(args) , s), None, loc)
    return loc['tmpfun']


## COLOR STUFF

color_ranges = {
    "rainbow": ((1,0,0),(1,1,0),(0,1,0),(0,1,1),(0,0,1),(1,0,1)),
    "revbow":  ((1,0,1),(0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0)),
    "fivecol": ((0,1,1),(0,1,0),(1,1,0),(1,0,0),(1,0,1)),
    }


def get_color(s):
    """Interpret argument as color or list of colors"""

    if type(s) == str:
        if s in color_ranges:
            return color_ranges[s]
        elif s.startswith("#"):
            col = int(s[1:],16)
            return ((col >> 16)/255., ((col >> 8) & 255)/255., (col & 255)/255.)
        elif s.isalpha():
            # A color name... This is PyMOL specific. Think of something better!
            return cmd.get_color_tuple(s)
        else:
            return get_color(ast.literal_eval(s))
    elif type(s) in (tuple,list):
        if len(s) == 3 and type(s[0]) in (int,float):
            return s
        else:
            return [ get_color(i) for i in s ]
    elif type(s) == int:
        # A Pymol color index? 
        return cmd.get_color_tuple(s)


def get_frac_color(x,colors,dcolor=None):
    """Return the color corresponding to point x between 0.0 and 1.0"""

    if type(colors[0]) not in (tuple,list):
        return colors
    if x <= 0:
        return colors[0]
    y   = x*(len(colors)-1)
    if y >= len(colors)-1:
        return colors[-1]
    idx = int(y)
    d   = y - idx
    col1 = colors[idx]
    if dcolor:
        return [i+d*j for i,j in zip(colors[idx],dcolor[idx])]
    return [i+d*(j-i) for i,j in zip(colors[idx],colors[idx+1])]


def colored_segments(breaks,radii,colors):
    """Split cylinder/cone segments defined by breaks further to match breaks in color range"""

    # radius per segment: cylinders
    # radius per break:   cones

    nbreaks    = len(breaks)
    nsegments  = nbreaks-1
    ncolors    = len(colors)

    if type(radii) in (float,int):
        radii = nsegments*[radii]
    if len(radii) == nbreaks:
        radii = list(zip(radii, radii[1:])) # cone radii per segment
    else:
        radii = list(zip(radii, radii))     # cylinder radii per segment
    
    bmin, bmax = min(breaks), max(breaks)
    brange     = [ ((i-bmin)/(bmax-bmin),i) for i in breaks ]
    crange     = [ (float(i)/(ncolors-1),j) for i,j in enumerate(colors) ]

    # for each segment
    # break segment at color breaks (mind radius)
    # break color at segment break
    # we start with both at 0
    # if s[0].end > c[1]: break segment at c[1]   -> s,e,rs,re,c[0],c[1]; pop c[0], set s[0].start
    # if s[0].end < c[1]: break color at s[0].end -> s,e,rs,re,c[0],ce;   set c[0], pop s[0]
    out   = []
    while len(brange) > 1:
        bs,sb = brange[0]
        be,eb = brange[1]
        cs,sc = crange[0]
        ce,ec = crange[1]
        rs,re = radii[0] 
        if be > ce:
            # break segment at color break
            f = (ce-bs)/(be-bs)
            r = rs+f*(re-rs) if re != rs else rs
            b = sb+f*(eb-sb)
            out.append(((sb,b),(rs,r),(sc,ec)))
            crange.pop(0) # End color is the new start color
            brange[0] = (ce,b)
            radii[0] = (r,re)
        elif be < ce:
            # break color at segment break            
            f = (be-cs)/(ce-cs)
            c = [ i+f*(j-i) for i,j in zip(sc,ec) ] # interpolate colors
            out.append(((sb,eb),(rs,re),(sc,c)))
            crange[0] = (be,c)
            brange.pop(0)
            radii.pop(0)
        else:
            # coinciding breaks
            out.append(((sb,eb),(rs,re),(sc,ec)))
            brange.pop(0)
            crange.pop(0)
            radii.pop(0)

    return out
                        

## CGO shapes    

def cgo_cone(start, end, r1=0.2, r2=0.0, col1=(1,1,1), col2=(1,1,1), n1=1.0, n2=1.0):
    """Auxiliary function for drawing a simple/single CGO cone"""
    return [27.0, start[0], start[1], start[2], end[0], end[1], end[2], r1, r2,
            col1[0], col1[1], col1[2], col2[0], col2[1], col2[2], n1, n2]


def cgo_cylinder(start, end, r, col1, col2):
    return [9.0, start[0], start[1], start[2], end[0], end[1], end[2],
            r, col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]


def cgo_sausage(start, end, r, col1, col2):
    return [14.0, start[0], start[1], start[2], end[0], end[1], end[2],
            r, col1[0], col1[1], col1[2], col2[0], col2[1], col2[2]]


def cgo_segments(start,end=None,delta=None,segments=None,**kwargs):
    """Generate a series of colored cylinders/sausages/cones"""

    if delta == None:
        delta = end-start

    rscale  = kwargs.get("rscale",1)
    obj = []

    # scaling and offset for cylinder segments
    # this makes sense for histogram/density
    binscale = float(kwargs.get("binscale",1.0))
    offset   = float(kwargs.get("offset",0.0))
    
    lo, up = 0, len(segments)
    if kwargs.get("sausage",False):
        (s,e),(r1,r2),(c1,c2) = segments[0]
        s, e = s+offset*(e-s), s+(offset+binscale)*(e-s)
        if r1 == r2:
            obj.append(cgo_sausage(start+s*delta,start+e*delta,rscale*r1,c1,c2))
            lo = 1
        (s,e),(r1,r2),(c1,c2) = segments[-1]
        s, e = s+offset*(e-s), s+(offset+binscale)*(e-s)
        if r1 == r2:
            obj.append(cgo_sausage(start+s*delta,start+e*delta,rscale*r1,c1,c2))
            up = -1
            
    for (s,e),(r1,r2),(c1,c2) in segments[lo:up]:
        s, e = s+offset*(e-s), s+(offset+binscale)*(e-s)
        if r1==r2:
            obj.append(cgo_cylinder(start+s*delta,start+e*delta,rscale*r1,c1,c2))
        else:
            obj.append(cgo_cone(start+s*delta,start+e*delta,rscale*r1,rscale*r2,c1,c2))
    return obj

      
##


class Histogram:
    def __init__(self,data,binsize=1.0,bins=0,normalize=False,convolve=None,weight=None):
        """Get the bin starts and counts of the values in vector"""
 
        if not isinstance(data,numpy.ndarray):
            data = numpy.array(data)

        # Divide by bin size (but shifted half a bin size to have one bin around 0)
        if bins:
            binsize = (data.max()-data.min())/bins
        scaled  = data/binsize+0.5

        # Truncate
        binned  = numpy.floor(scaled).astype('int')

        # Shift to have first bin at zero
        minbin  = binned.min() 
        binned -= minbin

        # Count number per bin 
        count   = numpy.bincount(binned)
        if weight is not None:
            weighed  = numpy.bincount(binned,weight)
            weighed2 = numpy.bincount(binned,weight**2)
            fcount   = count.astype("float32")
            # NOTE: this may introduce NaNs
            self.average = weighed/fcount
            self.var     = (weighed2-fcount*(weighed**2))/(fcount-1)

        if convolve:
            if type(convolve) == str:
                convolve = ast.literal_eval(convolve)
            count = numpy.convolve(count,convolve)

        # Get the start and end coordinate per bin
        starts  = (numpy.arange(count.size+1)+minbin-0.5)*binsize
    
        self.count = count
        self.bins  = starts


## Nipals

class Component:
    def __init__(self,loadings,scores,nipit):
        self.loadings = loadings.reshape((-1,3))
        self.scores   = scores
        self.mean     = nipit.mean.reshape((-1,3))
        self.sele     = nipit.selections[0]
        self.name     = nipit.name

    def _scaling(self, scale=1):
        if type(scale) == str:
            scale = scale.lower()
            if ("minmax".startswith(scale) or "extreme".startswith(scale)
                or "extent".startswith(scale) or "range".startswith(scale)):
                lower = self.scores.min()
                upper = self.scores.max()
                scale = 1
            elif ("sdev".startswith(scale) or "stdev".startswith(scale)
                  or "sigma".startswith(scale)):
                upper = self.scores.std()
                lower = -upper
                scale = 1
            elif "iqr".startswith(scale):
                upper, lower = numpy.percentile(self.scores,[25,75])
                scale = 1
            elif scale.endswith("%"):
                p = 0.5*float(scale[:-1])
                lower, upper = numpy.percentile(self.scores,[50-p,50+p])
                scale = 1
            else:
                upper = self.scores.std()
                lower = -upper
                scale = ast.literal_eval(scale)
        else:
            # Scaling by n times the standard deviation
            upper = self.scores.std()
            lower = -upper
        return scale, lower, upper


    def quantile(self,q=50,name=None):
        """Return the q'th quantile structure of the component""" 
        
        if not name:
            name = self.name+"_q%f"%q

        mod = cmd.get_model(self.sele)
        q, q25, q75 = numpy.percentile(self.scores,(q,25,75))
        for i,j,k in zip(mod.atom, self.mean, self.loadings):
            i.coord = (j+q*k).tolist()
            i.b     = numpy.sqrt(numpy.sum(((q75-q25)*k)**2))
        cmd.load_model(mod,name)


    def median(self,name=None):
        if not name:
            name = self.name + "_median"
        self.quantile(q=50,name=name)


    def cgo(self, **kwargs):
        """Return CGO representation of component"""

        ## Extract 'here'-options from kwargs
        what      = kwargs.pop("draw","boxplot").lower()
        radius    = kwargs.pop("radius",0.25)
        scale     = kwargs.pop("scale",1)
        reverse   = 1 if kwargs.pop("reverse",False) else 0
        binsize   = kwargs.pop("binsize",1.0)
        bins      = kwargs.pop("bins",None)
        convolve  = kwargs.pop("convolve",None)
        threshold = kwargs.get("threshold",-1e8)

        ## (Length) scaling for cylinders/(bi)cones
        #  minmax, extent, extreme, range, sd, sdev, stdev, sigma, iqr, 95%
        #  for boxplots/histograms this is reset to extent below
        scale, lower, upper = self._scaling(scale)

        ## Set colors
        histogram  = None
        color_type = "normal"
        color      = kwargs.get("color",(1,1,1))
        if type(color) == str and ":" in color:
            color_type, color = color.split(':')
        print(color_type, color)
        color      = get_color(color)
        if type(color[0]) in (float,int):
            color  = [color,color]
        if color_type in ("time","order","density","tmvar"):
            weights   = numpy.arange(len(self.scores))
            histogram = Histogram(self.scores,binsize,bins,weight=weights)
            density   = histogram.count/float(max(histogram.count))
            if color_type == "density":
                color = [ get_frac_color(i,color) for i in density ]
            else:
                if color_type == "tmvar":
                    var = histogram.var/max(histogram.var)
                    var[~numpy.isfinite(var)]  = -1
                    color = [ get_frac_color(i,color) for i in var ]
                else:
                    aver  = histogram.average/weights.max()
                    aver[~numpy.isfinite(aver)] = -1
                    color = [ get_frac_color(i,color) for i in aver ]


        ## Check for sausages
        sausage = "sausage".startswith(what)
        kwargs["sausage"] = sausage

        atomloads = numpy.sqrt((self.loadings**2).sum(axis=1))

        ## (Radial) scaling
        #  "loadings", a value, or a list of values (per atom)
        print(type(radius))
        if type(radius) == str:
            if "loadings" in radius:
                loads   = atomloads/atomloads.max()
                radius  = radius.replace("loadings","numpy.array(loadings)")
                radius  = parse_keyword_function(radius,"loadings")(loads.tolist())
            else:
                radius  = ast.literal_eval(radius)

        ## Select what to do
        if "density".startswith(what) or "histogram".startswith(what):
            lower, upper = min(self.scores), max(self.scores)
            histogram    = histogram or Histogram(self.scores,binsize,bins,convolve=convolve)
            points       = (histogram.bins-lower)/(upper-lower)
            maxcount     = float(max(histogram.count))
            segments     = colored_segments( points, histogram.count/maxcount, color)

        elif "boxplot".startswith(what) or "fivenum".startswith(what):
            lower, upper = min(self.scores), max(self.scores)
            whisk        = kwargs.pop("whisk_scale",0.25)
            band         = kwargs.pop("band_scale",1.25)
            points       = (numpy.percentile(self.scores, [0,25,47.5,52.5,75,100])-lower)/(upper-lower)
            segments     = colored_segments(points, [whisk,1,band,1,whisk], color)

        elif "cones".startswith(what) or "porcupine".startswith(what):
            segments     = colored_segments([0,1], [1-reverse,reverse], color)

        elif "cylinders".startswith(what) or sausage:
            segments     = colored_segments([0,1], 1, color)

        elif "bicone".startswith(what):
            segments     = colored_segments([0,0.5,1], [reverse,1-reverse,reverse], color)

        else:
            raise ValueError("Unknown viewing mode for loadings (what): %s"%what)

        ## Determine the coordinates of control points
        lower   = self.mean + scale*lower*self.loadings
        delta   = scale*(upper-lower)*self.loadings 
        upper   = self.mean + scale*upper*self.loadings
        
        ## Build CGO object
        obj = []
        if hasattr(radius,"__iter__"):
            for start,end,rscale,ld in zip(lower,upper,radius,atomloads):
                if ld > threshold:
                    obj.append(cgo_segments(start,end,segments=segments,rscale=rscale,**kwargs))
        else:
            for start,end,ld in zip(lower,upper,atomloads):
                if ld > threshold:
                    obj.append(cgo_segments(start,end,segments=segments,rscale=radius,**kwargs))

        ## Draw CGO object
        if "name" in kwargs:
            cmd.load_cgo(unpack(obj), kwargs["name"])

        return obj


## PCA

class NipalsIterator:
    """Iterator giving loadings and scores one at a time using Nipals(1) algorithm"""

    def __init__(self,X,scale=False):
        """Initialize Nipals(1) for matrix X with n variables (rows) on k observations (columns).""" 
        self.X = X

        # Mean structure as vector
        self.mean = self.X.mean(axis=0)

        # Deviations from mean
        self.X -= self.mean

        # Mean structure as 3D point set
        self.mean = self.mean.reshape((-1,3))

        # Variances
        self.var = self.X.var(axis=0)

        # Scale to unit variance (correlations)
        if scale:
            self.X  /= numpy.sqrt(self.var)
            self.var = numpy.ones(self.var.shape)
 
        # Total variance
        self.tot = self.var.sum()

        # Number of states
        self.nstates, self.natoms = self.X.shape
        self.natoms /= 3

        print("Got %d atoms in %d states. Total variance: %.5f" % (self.natoms, self.nstates, self.tot))

        # Prepare result arrays
        self.loadings  = []
        self.scores    = []
        self.values    = []
        self.explained = []


    def __len__(self):
        return len(self.loadings)


    def __getitem__(self,idx):
        if type(idx) != int:
            raise TypeError("Component indices should be integers.")
        return Component(self.loadings[idx], self.scores[idx], self)


    def __iter__(self):
        return self


    def next(self):
        if len(self.values) > min(self.nstates,self.natoms):
            raise StopIteration        
        start = time.time()
        self.nipals()
        print("Eigenvector %d, eigenvalue: %.5f (%.2f%%), cumulative: %.5f (%.2f%%) (%ds)"%(
            len(self.values), self.values[-1], 100*self.values[-1]/self.tot,
            sum(self.values), 100*sum(self.values)/self.tot, time.time()-start))
        return Component(self.loadings[-1], self.scores[-1], self)


    def nipals(self):
        """One NIPALS cycle, giving one loadings vector and one score vector"""

        scores   = self.X[:,self.X.var(axis=0).argmax()]          
        prev     = 0*scores
        loadings = None

        while numpy.sum((scores-prev)**2) > _R2SMALL:
            loadings = normalized(numpy.dot(self.X.T,scores.T))   
            prev     = scores                                     
            scores   = numpy.dot(self.X,loadings)

        if loadings is None:
            raise StopIteration

        # Model part from the current component, subtracted from data
        M = numpy.outer(scores,loadings)
        self.X -= M

        self.values.append(numpy.var(scores))
        self.scores.append(scores)
        self.loadings.append(loadings)
        self.explained.append((M**2).sum(axis=0)/self.var)
        
    
    def atomvar(self):
        return self.var.reshape((len(self.var)//3,3)).sum(axis=1)


    def write(self,name,what,**kwargs):
        """Write <what> to a file as matrix"""
        numpy.savetxt(fname, getattr(self,"what"), kwargs.get('fmt','%.18e'))


    def tofile(self, **kwargs):
        """Write data to files"""

        for what in ("loadings","scores"):
            filename = kwargs.pop(what,None)
            if what:
                self.write_loadings(filename,what,**kwargs)

        residuals = kwargs.pop("residuals",None)
        if residuals:
            self.write(residuals,"X",**kwargs)
                

    def cgo(self,**kwargs):
        """Return a CGO object for the given component"""

        comp = kwargs.pop("comp",-1)
        while comp > -1 and len(self.loadings) < comp:
            self.next()
        return self[comp].cgo(**kwargs)


## VARIMAX

def varimax(components, gamma=1, maxiter=20, tol=1e-8):
    """Perform VariMax rotation (gamma=1) on components"""
    p,k = components.shape
    R   = numpy.eye(k)
    f   = float(gamma)/p
    d   = 0
    for i in xrange(maxiter):
        d_old = d
        L = numpy.dot(components, R)
        A = L**3 - f * (L*(L**2).sum(axis=0))**2
        U,s,V = numpy.linalg.svd(numpy.dot(components.T,A))
        R = numpy.dot(U,V)
        d = sum(s)
        if (d - d_old)**2 < tol: 
            break
    return numpy.dot(components, R)


class Varimax:
    def __init__(self,components,**kwargs):
        if type(components) == int:
            self._do_pca(components,**kwargs)
        elif isinstance(components,numpy.ndarray):
            self._from_array(components,**kwargs)
        elif type(components) in (list,tuple) and isinstance(components[0],Component):
            self._from_components(components,**kwargs)
        else:
            self._from_pca(components,**kwargs)

    def _from_pca(self,pca,**kwargs):
        NotImplemented


## PYMOL SPECIFIC ROUTINES ##

_as_program      = (__name__ == "__main__")
_as_pymol_script = (__name__ == "pymol")
_as_module       = not (_as_program or _as_pymol_script)
_as_pymol_module = False

try:
    import pymol
    from pymol import cmd,cgo   
    import ast
    
    if _as_module:
        _as_pymol_module = True
        print("Importing PCA module.")
    else:
        print("Running PCA module.")

except ImportError:
    cmd = None
    _as_pymol_script = False


#
# Decorators
# ==========
#
# Two decorators are defined in this module, which allow 
# defining functions specific to a given environment 
# (run as program or as PyMOL script). If a function is 
# accessed outside of its scope, a message will be printed. 
# An alternative would be defining functions inside if-clauses, 
# but this way the level of indentation is kept lower.
#


def pymol_function(function):
    """Decorator to restrict functions to Pymol scope"""
    def wrapper(*args, **kwargs):
        if _as_pymol_script or _as_pymol_module:
            kwargs["_self"] = kwargs.get("_self",cmd)
            return function(*args,**kwargs)
        else:
            print("Function %s is only available when %s is run in Pymol.\n(%s)" % (function.__name__, __file__, function.__doc__))
    wrapper.__name__ = function.__name__
    wrapper.__doc__  = function.__doc__
    return wrapper


@pymol_function
def pymol_extend(function,_self=None):
    """Decorator to extend the Pymol interpreter"""
    _self.extend(function.__name__,function)
    return function
        

@pymol_function
def expand_selection(selection="all",_self=None):    
    """Split selection in objects, object-selections and states"""

    # Get object list of selection
    objects = _self.get_object_list(selection)

    # Set the selection string for specific objects
    selects = [ "%s and (%s)"%(i,selection) for i in objects ]

    # Get the number of states for each object
    nStates = [ _self.count_states(i) for i in objects]

    # Return the lists of objects, selections and states
    return objects, selects, nStates 


@pymol_function
def sele2array2D(selection, states=(0,), _self=None):
    """Read in structures into an array"""
    # This representation makes it very easy to get the 
    # inner products, simply from numpy.dot(X.T,X)
    # This gives a lot of overhead, since every Sij and Sji 
    # are calculated explicitly, but it is faster than 
    # managing things on the Python level.
    # A specific structure k can be extracted by:
    #     X.T.reshape(nsets,dim,natoms)[k,:,:].T
    objects = _self.get_object_list('(' + selection + ')')
    ostates = [ range(cmd.count_states(i)) for i in objects ]
    nobj    = len(objects)
    nstates = 0
    flat    = []
    for s in states:
        if type(s) in (list, tuple):
            if len(s) > 1:
                s = range(s[0],s[-1])
        else:
            s = [s]
        for k in s:
            lf = len(flat)
            _self.iterate_state(k, selection, '_append((x,y,z))',
                                space={'_append': flat.append}, atomic=0)
            if not k:
                nstates += sum(_self.count_states('%' + name) for name in objects)
            else:
                for o in ostates:
                    nstates += (k-1 in o)                    
    return numpy.reshape(flat, (nstates, -1))


class PymolNipalsIterator(NipalsIterator):
    def __init__( self, selection, states=0, name="pca", **kwargs):

        self.draw = kwargs.get("draw",False)
        if type(self.draw) == str and self.draw.lower() in ("n","no","none"):
            self.draw = False

        self.name       = name
        self.kwargs     = kwargs
        self.components = []

        # Get a list of all objects and all states per object
        obj, sel, sta   = expand_selection(selection)
        self.objects    = obj
        self.selections = sel
        self.states     = [ast.literal_eval(states)] if type(states) == str else (states,) 

        # Pass the coordinates to the superclass
        X = sele2array2D(selection,self.states)
        NipalsIterator.__init__(self,X)


    def next(self):
        # Determine next set of loadings and scores
        self.components.append(NipalsIterator.next(self))

        # Build a CGO plot
        if self.draw:
            cmd.load_cgo(unpack(self[-1].cgo(**self.kwargs)),self.name)


    def load_mean(self,name=None):
        if not name:
            name = self.name + "_mean"
        
        mod = cmd.get_model(self.selections[0])
        for i,j,k in zip(mod.atom,self.mean,self.var.reshape((len(self.var)//3,3))):
            i.coord = j.tolist()
            i.b     = k.sum()
        cmd.load_model(mod,name)


    def load_quantile(self,quantile=50,comp=-1,name=None):
        if not name:
            name = self.name + "_q%f"%quantile
        mod = cmd.get_model(self.selections[0])
        q, q25, q75 = numpy.percentile(self.scores[comp],(quantile,25,75))
        for i,j,k in zip(mod.atom, self.mean, self.loadings[comp].reshape(self.mean.shape)):
            i.coord = (j+q*k).tolist()
            i.b     = numpy.sqrt(numpy.sum(((q75-q25)*k)**2))
        cmd.load_model(mod,name)


    def load_median(self,comp=-1,name=None):
        if not name:
            name = self.name + "_median"
        self.load_quantile(q=50,comp=comp,name=name)


@pymol_extend
@pymol_function
def princomp(selection="all",states=(0,),name="pca",maxvec=1,**kwargs):
    """
    DESCRIPTION
    
            "princomp" performs principal component analysis on an ensemble
            of structures from multiple objects and/or states. Components are
            calculated using the Nipals algorithm (Wold 1987). This method
            is generally fast, but requires collecting the coordinates from
            all states, which may take some time. Components are drawn as
            CGO objects. Subsequent components can be determined using "nextcomp".

    USAGE 
    
            princomp selection, name, maxvec[, radius[, color[, what[, scale[, ...]]]]]
    
    ARGUMENTS
    
            selection = string: atom selection 
    
            name      = string: name for pca object
    
            draw      = string: display type of components, "cone" or "cylinder"

            radius    = float:  base radius of cylinders/cones

            color     = tuple:  color list/tuple (r,g,b) or multiple tuples ((r,g,b),...,(r,g,b))                                
            
            maxvec    = int:    number of components to determine

            scale     = float:  scale factor for drawing components

            binsize   = float:  size of bins for displaying densities

            convolve  = tuple:  smoothing window for densities

    NOTES
    
            "princomp" creates a PCA iterator object with the given 
            name in the global namespace. This object can be used
            to obtain subsequent components by calling its .next()
            method.
    
    EXAMPLE
    
            princomp not hydro, name=PCA, radius=0.15, maxvec=5
    
    SEE ALSO

            nextcomp

    """
    maxvec = int(maxvec)
    start  = time.time()
    PNI    = PymolNipalsIterator(selection,states=states,name=name,**kwargs)
    print("(%ds) Created PCA iterator %s. Determining first %d eigenvectors:" % (time.time()-start,name,maxvec))
    for i in range(maxvec):
        PNI.next()
    print('(%ds) Use "nextcomp %s" for next eigenvector.' % (time.time()-start,name))
    globals()[name] = PNI
    return PNI



@pymol_extend
@pymol_function
def nextcomp(obj="pca",_self=None):
    """
    DESCRIPTION
    
            "nextcomp" determines and draws the next eigenvector from a
            PCA object set up with "princomp"

    USAGE 
    
            nextcomp object
    
    ARGUMENTS
    
            object    = string: PCA object

    EXAMPLE
    
            nextcomp pca
    
    SEE ALSO

            princomp, drawmean

    """
    pca_obj = globals().get(obj)
    if not isinstance(pca_obj,NipalsIterator):
        print('There is no PCA object called "%s".'%obj)
        return
    pca_obj.next()
    

@pymol_extend
@pymol_function
def drawcomp(obj="pca", comp=-1, state=-1, name=None, **kwargs):
    """
    DESCRIPTION
    
            "drawcomp" draws a component from a PCA object 
            set up with "princomp"

    USAGE 
    
            drawcomp object, comp, name, what, radius, color, scale, binsize, convolve
    
    ARGUMENTS
    
            object    = string: PCA object

            name      = string: name for pca object
    
            what      = string: display type of components, "cone" or "cylinder"

            radius    = float:  base radius of cylinders/cones

            color     = tuple:  color list/tuple (r,g,b) or multiple tuples ((r,g,b),...,(r,g,b))                                
            
            scale     = float:  scale factor for drawing components

            binsize   = float:  size of bins for displaying densities

            convolve  = tuple:  smoothing window for densities

    EXAMPLE
    
            drawcomp pca, what=density, radius=10, color=((1,0,0),(1,1,1),(0,1,1)), convolve=[1,2,3,2,1]
    
    SEE ALSO

            princomp, nextcomp, drawmean

    """
    pca_obj = globals().get(obj)
    if not isinstance(pca_obj,NipalsIterator):
        print('There is no PCA object called "%s".'%obj)
        return

    state   = int(state)
    comp    = int(comp)
    if name == None:
        if comp < 0:
            comp = len(pca_obj) - 1
        name = pca_obj.name + "_pc%d"%comp
    cgo_obj = pca_obj[comp].cgo(**kwargs)
    cmd.load_cgo(unpack(cgo_obj), name, state)


@pymol_extend
@pymol_function
def drawmean(obj="pca",name=None,_self=None):
    """
    DESCRIPTION
    
            "drawmean" loads the mean structure from a PCA object 
            set up with "princomp"

    USAGE 
    
            drawmean object
    
    ARGUMENTS
    
            object    = string: PCA object

    EXAMPLE
    
            drawmean pca
    
    SEE ALSO

            princomp, nextcomp

    """
    pca_obj = globals().get(obj)
    if not isinstance(pca_obj,NipalsIterator):
        print('There is no PCA object called "%s".'%obj)
        return
    if not name:
        name = obj+"_mean"
    pca_obj.load_mean(name)
    

@pymol_extend
@pymol_function
def drawmedian(obj="pca", component=-1, name=None, _self=None):
    """
    DESCRIPTION
    
            "drawmedian" loads the median structure from a PCA object 
            set up with "princomp"

    USAGE 
    
            drawmedian object
    
    ARGUMENTS
    
            object    = string: PCA object

            component = int:    Component to draw median of
 
            name      = string: Name for resulting structure
    EXAMPLE
    
            drawmedian pca
    
    SEE ALSO

            princomp, nextcomp

    """
    pca_obj = globals().get(obj)
    if not isinstance(pca_obj,NipalsIterator):
        print('There is no PCA object called "%s".'%obj)
        return
    if not name:
        name = obj+"_median"
    component = ast.literal_eval(component) if type(component) == str else component
    pca_obj[component].median(name=name)
    

@pymol_extend
@pymol_function
def drawquantile(obj="pca", quantile=50, component=-1, name=None, _self=None):
    """
    DESCRIPTION
    
            "drawquantile" loads the structure from a PCA object 
            set up with "princomp" corresponding to the given quantile.

    USAGE 
    
            drawquantile object, quantile[, quantile[, name]]
    
    ARGUMENTS
    
            object    = string:                   PCA object

            quantile  = float or list of floats:  Quantiles, between 0.0 and 1.0
    
            component = index:                    Component to use (-1 is last)

            name      = string:                   Name for the resulting structure
  
    EXAMPLE
    
            drawquantile pca, (25, 75), name=IQR
    
    SEE ALSO

            princomp, nextcomp

    """
    pca_obj = globals().get(obj)
    if not isinstance(pca_obj,NipalsIterator):
        print('There is no PCA object called "%s".'%obj)
        return
    quantile = ast.literal_eval(quantile) if type(quantile) == str else quantile
    comp     = ast.literal_eval(component) if type(component) == str else component
    if not type(quantile) in (list,tuple):
        quantile = [quantile]
    for q in quantile:
        pca_obj[comp].quantile(q=float(q),name=name)


@pymol_extend
@pymol_function
def project(selection="all",obj="pca",comp=None,states=0,mean=False,name=None):
    """
    DESCRIPTION
    
            "project" projects a selection onto the components of a pca object

    USAGE 
    
            project object, selection, name
    
    ARGUMENTS
    
            obj       = string: PCA object

            selection = string: selection, matching the PCA object selection

            comp      = int or tuple of ints: 

            name      = string: name of object to create

    EXAMPLE
    
            project all, pca
    
    SEE ALSO

            princomp

    """
    states = [ast.literal_eval(states)] if type(states) == str else (states,)
    comp   = ast.literal_eval(comp)     if type(comp)   == str else comp
    mean   = ast.literal_eval(mean)     if type(mean)   == str else mean

    coords = sele2array2D(obj.selection + " and " + selection, states)
    if mean:
        coords = coords.mean(axis=1).reshape((-1,1))
    loads  = numpy.array(obj.loadings)[comp,:]
    scores = numpy.dot(loads, (coords-obj.mean.flatten()).T)
    coords = numpy.dot(loads.T, scores)

    mod = cmd.get_model(obj.selection + " and " + selection)
    for s in range(coords.shape[1]):
        for i,j,k in zip(mod.atom,obj.mean,coords[:,s].reshape((-1,3))):
            i.coord = j.tolist()
        cmd.load_model(mod,name)

    return scores