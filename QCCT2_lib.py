# -*- coding: utf-8 -*-
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Warning: THIS MODULE EXPECTS PYQTGRAPH DATA: X AND Y ARE TRANSPOSED!

Changelog:
    20210422: largest components for find cylinder; no blur for cylinder by default
    20210415: completely rewritten, for easier maintenance
    20210413: more comments; allow head and body swap; changes to allow rotated inserts
    20200508: dropping support for python2; dropping support for WAD-QC 1; toimage no longer exists in scipy.misc
    20200225: prevent accessing pixels outside image
    20171116: fix scipy version 1.0
    20170622: more extreme avg_skull value to allow reporting for body
    20170502: added radiusmm param for air roi location; added thumbnail with ROIs
    20161220: removed testing stuff; removed class variables
    20161216: allow manually supplied anatomy
    20160902: sync with wad2.0; Unified pywad1.0 and wad2.0
    20150701: updated for iPatient iCT: body and head; other tags; other body
    20150409: Removed scanner definitions; should be passed to cs or in config
    20141009: Update to use dicomMode instead of mode2D
    20140528: Initialize all items of CTStruct in __init__ (bug for gui)
    20140425: Bugfix FindCenterShift
    20140414: Removed 's in DICOM tag name to avoid sql problems
    20140409: Initial split of gui/lib for pywad

"""
"""
TODO:
 o scanner definition from params: phantom diam, inserts (materials, positions), materials
"""
__version__ = '20210422'
__author__ = 'aschilham'
TRANSPOSED_XY = False # normal!

## WAD-QC imports: BEGIN
LOCALIMPORT = False
try: 
    # try local folder
    import wadwrapper_lib
    LOCALIMPORT = True
except ImportError:
    # try wad2.0 from system package wad_qc
    from wad_qc.modulelibs import wadwrapper_lib

try:
    from scipy.misc import toimage
except (ImportError, AttributeError) as e:
    try:
        if LOCALIMPORT:
            from wadwrapper_lib import toimage as toimage
        else:
            from wad_qc.modulelibs.wadwrapper_lib import toimage as toimage
    except (ImportError, AttributeError) as e:
        msg = "Function 'toimage' cannot be found. Either downgrade scipy or upgrade WAD-QC."
        raise AttributeError("{}: {}".format(msg, e))
## WAD-QC imports: END

import copy
import numpy as np
from scipy import stats
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image # image from pillow is needed
from PIL import ImageDraw # imagedraw from pillow is needed, not pil

# sanity check: we need at least scipy 0.10.1 to avoid problems mixing PIL and Pillow
scipy_version = [int(v) for v in scipy.__version__ .split('.')]
if scipy_version[0] == 0:
    if scipy_version[1]<10 or (scipy_version[1] == 10 and scipy_version[1]<1):
        raise RuntimeError("scipy version too old. Upgrade scipy to at least 0.10.1")

def save_image_with_rois(image, fname, circle_rois=[], rect_rois=[], poly_rois=[]):
    """
    save a jpeg with indicated rois 
    """
    def _draw_thick_rectangle(draw, roi, color, thick):
        """
        helper function drawing a rectangular ROI.
        roi = [ [x0,y0], [x1,y1] ] or [ x0, y0, x1, y1 ]
        """
        #[(x0,y0),(x1,y1)]
        if len(roi) == 4:
            x0,y0,x1,y1 = roi
        else:
            x0,y0 = roi[0]
            x1,y1 = roi[1]

        for t in range(-int((thick-1)/2),int((thick+1)/2)):
            draw.rectangle([(x0+t,y0+t),(x1-t,y1-t)],outline=color)


    # make a palette, mapping intensities to greyscale
    pal = np.arange(0,256,1,dtype=np.uint8)[:,np.newaxis] * \
        np.ones((3,),dtype=np.uint8)[np.newaxis,:]
    # but reserve the first for red for markings
    pal[0] = [255,0,0]

    # convert to 8-bit palette mapped image with lowest palette value used = 1
    # first the base image
    im = toimage(image, low=1, pal=pal) 

    # now draw all rois in reserved color
    draw = ImageDraw.Draw(im)
    for r in poly_rois:
        roi =[]
        for x,y in r:
            roi.append( (int(x+.5),int(y+.5)))
        draw.polygon(roi, outline=0)

    for r in rect_rois:
        #draw.rectangle(r,outline=0)
        _draw_thick_rectangle(draw, r, 0, 3)

    # now draw all cirlerois in reserved color
    for x,y,r in circle_rois: # low contrast elements
        draw.ellipse((x-r,y-r,x+r,y+r), outline=0)
    del draw

    # convert to RGB for JPG, cause JPG doesn't do PALETTE and PNG is much larger
    im = im.convert("RGB")

    imsi = im.size
    if max(imsi)>2048:
        ratio = 2048./max(imsi)
        im = im.resize( (int(imsi[0]*ratio+.5), int(imsi[1]*ratio+.5)), Image.ANTIALIAS)
    im.save(fname)
    
class CTStruct:
    """
    structure that contains all results
    """
    def __init__ (self):
        # for matlib plotting
        self.hasmadeplots = False
        
        # phantom
        self.phantom_xcycdiampx = [] # xc, yc, diam in px of phantom

        # uniformity
        self.unif_roiavg = [] # Average HU in rois
        self.unif_roistd = [] # Standard Deviation HU in rois
        self.uniformity = -1. # non-uniformity

        # noise
        self.noise_roiavg = [] # Average HU in rois
        self.noise_roistd = [] # Standard Deviation HU in rois
        self.noise_roisnr = [] # (avg+1000)/sd

        # linearity
        self.lin_xcycdiampx = [] # xc, yc, diam in px of inserts
        self.lin_roiavg = [] # Average HU in rois
        self.lin_roistd = [] # Standard Deviation HU in rois
        self.lin_roiGT = [] # groundtruth HU values
        self.linearity_fit = -1. # fit
        self.linearity_dev = -1. # max deviation

        # for gui 
        self.gui_rois = [] # x0,y0,rad

        
class CT_QC:
    """
    class with CT specfic functions
    """
    def __init__(self, dcmInfile, pixeldataIn, dicomMode):
        self.qcversion = __version__

        # debug info or not
        self.verbose = False

        # input image
        self.dcmInfile = dcmInfile
        self.pixeldataIn = pixeldataIn
        self.dicomMode = dicomMode

        # prepare data
        if not pixeldataIn is None:
            self.work_im = self.prepare_workimage()

        # results
        self.results = CTStruct()
        
    def dicomtag_read(self, key, imslice=0): # slice=2 is image 3
        # wrapper to read 2D or 3D dicom tags
        value = wadwrapper_lib.readDICOMtag(key, self.dcmInfile, imslice)
        return value

    def dicomspacing_read(self):
        """
        Try to read it from proper DICOM field. If not available,
        calculate from distance between two slices, if 3D
        """
        key = "0018,0088" # "Spacing Between Slices", // Philips
        value = self.dicomtag_read(key)
        if self.dicomMode == wadwrapper_lib.stMode2D:
            return value
        if(value != ""):
            return value

        key = "0020,1041" #"Slice Location",
        val1 = self.dicomtag_read(key,imslice=0)
        val2 = self.dicomtag_read(key,imslice=1)
        if(val1 != "" and val2 != ""):
            value = val2-val1
        return value

    def pix2phantommm(self, pix):
        # translate pixels into mm
        if self.dicomMode == wadwrapper_lib.stMode2D:
            pix2mmm = self.dcmInfile.PixelSpacing[0]
        else:
            pix2mmm = self.dcmInfile.info.PixelSpacing[0]
        return pix*pix2mmm

    def phantommm2pix(self, mm):
        # translate mm into pixels
        if self.dicomMode == wadwrapper_lib.stMode2D:
            pix2mmm = self.dcmInfile.PixelSpacing[0]
        else:
            pix2mmm = self.dcmInfile.info.PixelSpacing[0]
        return mm/pix2mmm

    def prepare_workimage(self):
        """
        get the proper slice, remove table (exclude circle)
        """
        # set proper input slice
        if not self.dicomMode == wadwrapper_lib.stMode2D:
            recondiam = self.dcmInfile.info.ReconstructionDiameter # data reconstruction diameter
            dep = np.shape(self.pixeldataIn)[0]
            self.unif_slice = int((dep-1)/2) # take middle slice
            work_im = copy.deepcopy(self.pixeldataIn[self.unif_slice])
        else:
            recondiam = self.dcmInfile.ReconstructionDiameter # data reconstruction diameter
            self.unif_slice = 0 # there is only one slice
            work_im = copy.deepcopy(self.pixeldataIn)
        
        # remove table (well, remove circle)
        midpx  = int(self.phantommm2pix(recondiam/2.)+.5)
        work_im = self._select_circle(work_im, self.phantommm2pix(recondiam), (midpx,midpx) )
        return work_im
    
    def _select_circle(self, im, diam, xy0):
        """
        remove everything outside circle of diameter 'diam' around pixel center x,y=xy0
        """
        dimy,dimx = np.shape(im)
        # Define disc at roi location
        roirad = diam/2.
        x0,y0 = xy0

        # build x,y coordinates 
        if TRANSPOSED_XY:
            x,y = np.meshgrid( np.arange(0,dimy), np.arange(0,dimx), indexing='ij' )
        else:
            x,y = np.meshgrid( np.arange(0,dimy), np.arange(0,dimx), indexing='xy' )

        selection = np.zeros((dimy,dimx), dtype=np.bool)
        selection[((x-x0)**2.+(y-y0)**2.)<=roirad**2.] = True
        
        im[~selection] = -1000
        return im

    def _getlargestcc(self, segmentation):
        """
        https://stackoverflow.com/questions/47540926/get-the-largest-connected-component-of-segmentation-image
        return image with only label of largest component
        """
        from skimage.measure import label
        labels = label(segmentation)
        unique, counts = np.unique(labels, return_counts=True)
        list_seg = list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
        largest = max(list_seg, key=lambda x:x[1])[0]
        labels_max = (labels == largest).astype(int)
        return labels_max
        
    def _find_cylinder(self, im, plug, title=None):
        """
        find the best center for a cylinder
        plug = {'thresh_hi': 100, 'thresh_lo': None, 'sigma': 7.0}
        """
        from skimage.measure import regionprops,label

        error = True

        shiftxypx = [0,0]
        #dscale = 7.0 # blursigma_px

        dscale = plug.get('sigmapx', None)
        thresh_hi = plug.get('thresh_hi', None)
        thresh_lo = plug.get('thresh_lo', None)

        # 1. blur object
        if not dscale is None:
            blurIm = scipy.ndimage.gaussian_filter(im, sigma=dscale)
        else:
            blurIm = im
            
        # threshold image
        selection = np.ones(blurIm.shape, dtype=int)
        if not thresh_hi is None:
            selection[(blurIm >= thresh_hi)] = 0
        if not thresh_lo is None:
            selection[(blurIm <= thresh_lo)] = 0

        # keep only largest component
        selection = self._getlargestcc(selection)
        stats = regionprops(selection)[0] # ask for properties of first label (there is only one)
        cy,cx = stats.centroid # centroid of center of mass
        minaxislength = stats.minor_axis_length # smallest diam within mask
        shiftxypx = [cx, cy]
        if self.verbose:
            plt.figure()
            plt.title("find cylinder {}".format(title if not title is None else ""))
            plt.imshow(im, cmap=plt.cm.gray)
            cmap = copy.copy(plt.cm.jet) #copy.copy(mpl.cm.get_cmap("jet"))
            mask_opt = {'vmin':0.1, 'alpha':0.5, 'cmap': cmap}#plt.cm.jet}
            mask_opt["cmap"].set_under(alpha=0)
            plt.imshow(selection, **mask_opt)
                
            plt.plot([cx, cx-minaxislength/2, cx, cx+minaxislength/2, cx], 
                     [cy, cy, cy+minaxislength/2, cy, cy-minaxislength/2], 'ro')
            self.hasmadeplots = True
            
        error = False
        shiftxypx.append(minaxislength)
        return error,shiftxypx

    def _roi_stats(self, im, roi, title=None):
        """
        calculate average and stdev of given roi of given diam around given xy (all in px)
        roi = {'type': "disc", "xc": xc, "yc":yc, "diam":diam} all in pix
        """
        dimy, dimx = np.shape(im)

        if roi['type'] == "disc":
            # Define disc at roi location
            roirad = roi["diam"]/2.
            
            # build x,y coordinates 
            if TRANSPOSED_XY:
                x,y = np.meshgrid( np.arange(0,dimy), np.arange(0,dimx), indexing='ij' )
            else:
                x,y = np.meshgrid( np.arange(0,dimy), np.arange(0,dimx), indexing='xy' )
    
            selection = np.zeros((dimy,dimx), dtype=np.bool)
            selection[((x-roi["xc"])**2.+(y-roi["yc"])**2.)<=roirad**2.] = True
        elif roi['type'] == "ring":
            # a ring
            # Define disc at roi location
            roiradmin = roi["diam_min"]/2.
            roiradmax = roi["diam_max"]/2.
            
            # build x,y coordinates 
            if TRANSPOSED_XY:
                x,y = np.meshgrid( np.arange(0,dimy), np.arange(0,dimx), indexing='ij' )
            else:
                x,y = np.meshgrid( np.arange(0,dimy), np.arange(0,dimx), indexing='xy' )
    
            selection = np.zeros((dimy,dimx), dtype=np.bool)
            selection[((x-roi["xc"])**2.+(y-roi["yc"])**2.)<=roiradmax**2.] = True
            selection[((x-roi["xc"])**2.+(y-roi["yc"])**2.)<=roiradmin**2.] = False
 
 
        avg = np.ma.array(im, mask=~selection).mean()
        std = np.ma.array(im, mask=~selection).std()

        if self.verbose: 
            #im[selection] = -1000 # debug
            plt.figure()
            plt.title("ROI stats {}".format(title if not title is None else ""))
            plt.imshow(im, cmap=plt.cm.gray)
            cmap = copy.copy(plt.cm.jet) #copy.copy(mpl.cm.get_cmap("jet"))
            mask_opt = {'vmin':.1, 'alpha':0.5, 'cmap': cmap}#plt.cm.jet}
            mask_opt["cmap"].set_under(alpha=0)
            overlay = selection.astype(np.int8)
            plt.imshow(overlay, **mask_opt)
            self.hasmadeplots = True

        return avg, std

    def _pearson_coef(self, yarr, xarr):
        """
        Calcualate goodness of linear fit
        """
        r1_value = 0
        nonidentical = True
        ssxm, ssxym, ssyxm, ssym = np.cov(xarr,yarr, bias=1).flat
        if(ssxm == 0):
            nonidentical = False
        if(nonidentical):
            slope, intercept, r1_value, p_value, std_err = stats.linregress(xarr,yarr)
        return r1_value**2

    def run(self, pars):
        """
        run all analysis
        """
        error = True
        msg = ""

        # uniformity
        error = self.qc_uniformity(pars)
        if error:
            msg = "Error in Uniformity"
            return error, msg

        # noise
        error = self.qc_noise(pars)
        if error:
            msg = "Error in Noise"
            return error, msg

        # linearity
        error = self.qc_linearity(pars)
        if error:
            msg = "Error in Linearity"
            return error, msg

        return error, msg

    def qc_uniformity(self, pars):
        """
        Calculate uniformity only.
        pars = {
            'phantom': {'thresh_lo':-250, 'sigmapx':7.0},#, 'thresh_hi': 500}# HU, px
            'unifrois': [
                #roidimIEC   = int(self.phantommm2pix(cs,30.*datadiam/350)+.5) # 25 for body // IEC say at least 10% of diam (178mm)
                {'radmm':   0., 'angdeg':   0, 'diammm': 30},
                {'radmm': 120., 'angdeg': -90, 'diammm': 30},
                {'radmm': 120., 'angdeg': -45, 'diammm': 30},
            ]
        }
        pars['phantom'] is used to find the shifted location of the phantom
        pars['unifrois'] is used to calculate the max difference (in HU) between the first ROI (center) and the other ones
        """
        # 1. find shifted center of phantom
        phantom = pars['phantom']
        if self.results.phantom_xcycdiampx == []:
            error,self.results.phantom_xcycdiampx = self._find_cylinder(self.work_im, phantom, title="phantom")
        else:
            error = False
        if error:
            print("[uniformity] cannot find phantom center")
            return error

        # 2. build rois and calculate avg, std
        rois = pars['unifrois']
        self.results.unif_roiavg = []
        self.results.unif_roistd = []
        for i,roi in enumerate(rois):
            radpx  = self.phantommm2pix(roi['radmm'])
            angrad = np.deg2rad(roi['angdeg'])
            diampx = self.phantommm2pix(roi['diammm'])
            
            # center of roi
            x0 = self.results.phantom_xcycdiampx[0]+radpx*np.cos(angrad)
            y0 = self.results.phantom_xcycdiampx[1]+radpx*np.sin(angrad)
            
            # calc avg
            avg, std = self._roi_stats(self.work_im, {'type': "disc", "xc": x0, "yc":y0, "diam":diampx},
                                           title="uniformity {}".format(i))
            self.results.unif_roiavg.append(avg)
            self.results.unif_roistd.append(std)

            # for gui
            self.results.gui_rois.append([x0, y0, diampx/2.])

        # uniformity
        maxdev = abs(self.results.unif_roiavg[0]-self.results.unif_roiavg[1])
        for k in range(2,3):
            maxdev = max(maxdev, abs(self.results.unif_roiavg[0]-self.results.unif_roiavg[k]))
        self.results.uniformity = -maxdev
        
        return error

    def qc_noise(self, pars):
        """
        Calculate noise only.
        pars = {
            'phantom': {'thresh_lo':-250, 'sigmapx':7.0},#, 'thresh_hi': 500}# HU, px
            'noiserois': [
                #body: roidimIEC   = int(self.phantommm2pix(cs,30.*datadiam/350)+.5) # 25 for body // IEC say at least 10% of diam (178mm)
                #body: sddimIEC    = int(3.25*roidimIEC)#IEC say at least 40% of diam (178mm) and may not overlap with other structures
                {'radmm':   0., 'angdeg':   0, 'diammm': 30*3.25},
            ]
        }
        pars['phantom'] is used to find the shifted location of the phantom
        pars['noiserois'] is used to calculate the max difference (in HU) between the first ROI (center) and the other ones
        """
        # 1. find shifted center of phantom
        phantom = pars['phantom']
        if self.results.phantom_xcycdiampx == []:
            error,self.results.phantom_xcycdiampx = self._find_cylinder(self.work_im, phantom, title="phantom")
        else:
            error = False
        if error:
            print("[uniformity] cannot find phantom center")
            return error

        # 2. build rois and calculate avg, std
        rois = pars['noiserois']
        self.results.noise_roiavg = []
        self.results.noise_roistd = []
        for i,roi in enumerate(rois):
            radpx  = self.phantommm2pix(roi['radmm'])
            angrad = np.deg2rad(roi['angdeg'])
            diampx = self.phantommm2pix(roi['diammm'])
            
            # center of roi
            x0 = self.results.phantom_xcycdiampx[0]+radpx*np.cos(angrad)
            y0 = self.results.phantom_xcycdiampx[1]+radpx*np.sin(angrad)
            
            # calc avg and stdev
            avg, std = self._roi_stats(self.work_im, {'type': "disc", "xc": x0, "yc":y0, "diam":diampx},
                                           title="noise {}".format(i))
            self.results.noise_roiavg.append(avg)
            self.results.noise_roistd.append(std)

            # for gui
            self.results.gui_rois.append([x0, y0, diampx/2.])

        # SNR
        self.results.noise_roisnr = [(avg+1000.)/sd for avg,sd in zip(self.results.noise_roiavg, self.results.noise_roistd)]
        
        return error

    def qc_linearity(self, pars):
        """
        Calculate linearity only.
        pars = {
            'phantom': {'thresh_lo':-250, 'sigmapx':7.0},#, 'thresh_hi': 500}# HU, px
            'linrois': [
                #roidimMAT   = int(self.phantommm2pix(cs,18.*datadiam/350)+.5) # 2non official, like for uniformity and outside
                {'radmm':    0., 'angdeg':   0, 'diammm': 25., 'HU': 0}, # water
                {'radmm':   75., 'angdeg':   0, 'diammm': 25., 'thresh_lo':250, 'sigmapx':7.0, 'cutdiammm': 55, 'HU': 923}, # teflon
                {'radmm':    0., 'angdeg':   0, 'diam_minmm': 295., 'diam_maxmm': 299., 'HU': 120}, # between rad_in and rad_out # pvc
                {'radmm':  185., 'angdeg': -45, 'diammm': 25., 'HU': -1000}, # air
            ]
        }
        pars['phantom'] is used to find the shifted location of the phantom
        pars['linrois'] is used to calculate the max difference (in HU) between the first ROI (center) and the other ones
        """
        # 1. find shifted center of phantom
        phantom = pars['phantom']
        if self.results.phantom_xcycdiampx == []:
            error,self.results.phantom_xcycdiampx = self._find_cylinder(self.work_im, phantom, title="phantom")
        else:
            error = False
        if error:
            print("[linearity] cannot find phantom center")
            return error
        
        # for bound checks
        hei,wid = self.work_im.shape 

        # 2. build rois and calculate avg, std
        rois = pars['linrois']
        self.results.lin_roiavg = []
        self.results.lin_roistd = []
        for i,roi in enumerate(rois):
            # center of roi
            radpx  = self.phantommm2pix(roi.get('radmm',0.))
            angrad = np.deg2rad(roi.get('angdeg',0.))
            x0 = self.results.phantom_xcycdiampx[0]+radpx*np.cos(angrad)
            y0 = self.results.phantom_xcycdiampx[1]+radpx*np.sin(angrad)

            cutdim = roi.get('cutdiammm', None)
            diam_min = roi.get('diam_minmm', None)
            diam_max = roi.get('diam_maxmm', None)
            if not (None in [diam_min, diam_min]):
                # defined as 2 circles
                diam_minpx  = self.phantommm2pix(diam_min)
                diam_maxpx  = self.phantommm2pix(diam_max)
                avg, std = self._roi_stats(self.work_im, {'type': "ring", "xc": x0, "yc":y0, "diam_min":diam_minpx, "diam_max": diam_maxpx},
                                           title="linearity {}".format(i))
                xcycdiampx = [x0, y0, (diam_minpx+diam_minpx)/2.]
                # for gui
                self.results.gui_rois.append([x0, y0, diam_minpx/2.])
                self.results.gui_rois.append([x0, y0, diam_maxpx/2.])
            else:
                # some disc
                diampx = self.phantommm2pix(roi['diammm'])
                
                if not cutdim is None:
                    # locate insert: make a small cut-out and look for it
                    cutdim = int(self.phantommm2pix(self.phantommm2pix(roi['cutdiammm']))) # diam of cut-out
                    xstart = int(x0-cutdim/2)
                    xend   = xstart+cutdim
                    ystart = int(y0-cutdim/2)
                    yend   = ystart+cutdim
                    if xstart<0 or xend>wid-1 or ystart<0 or yend>hei-1:
                        raise ValueError("Trying to cut a region outside of the image for linearity ROI {}:(x{}-{}), (y{}-{})".format(i, xstart, xend, ystart, yend))

                    cut_im = self.work_im[ystart:yend,xstart:xend]
                    if self.verbose:
                        plt.figure()
                        plt.title("cut-out linearity {}".format(i))
                        plt.imshow(cut_im, cmap=plt.cm.gray)
        
                    error,xcycdiampx = self._find_cylinder(cut_im, roi, title="linearity {}".format(i))
                    
                    # calc avg and stdev
                    avg, std = self._roi_stats(self.work_im, {'type': "disc", "xc": xstart+xcycdiampx[0], "yc":ystart+xcycdiampx[1], "diam":diampx},
                                           title="linearity {}".format(i))
                    xcycdiampx = [xstart+xcycdiampx[0], ystart+xcycdiampx[1], diampx]
                else:
                    # normal disc with thresholds
                    # calc avg and stdev
                    avg, std = self._roi_stats(self.work_im, {'type': "disc", "xc": x0, "yc":y0, "diam":diampx},
                                           title="linearity {}".format(i))
                    xcycdiampx = [x0, y0, diampx]

            self.results.lin_roiavg.append(avg)
            self.results.lin_roistd.append(std)
            self.results.lin_roiGT.append(roi['HU'])
            self.results.lin_xcycdiampx.append(xcycdiampx)

            # for gui
            if None in [diam_min, diam_min]:
                self.results.gui_rois.append([xcycdiampx[0], xcycdiampx[1], xcycdiampx[2]/2.])


        # linearity
        self.results.linearity_fit = self._pearson_coef(self.results.lin_roiavg, self.results.lin_roiGT)
        maxdev = self.results.lin_roiavg[0]-self.results.lin_roiGT[0]
        for m,g in zip(self.results.lin_roiavg, self.results.lin_roiGT):
            if(abs(m-g)>abs(maxdev)):
                maxdev = m-g
        self.results.linearity_dev = maxdev

        return error
    
#----------------------------------------------------------------------
#----------------------------------------------------------------------
    def dicom_info(self,cs,info='dicom'):
        # Different from ImageJ version; tags "0008","0104" and "0054","0220"
        #  appear to be part of sequences. This gives problems (cannot be found
        #  or returning whole sequence blocks)
        # Possibly this can be solved by using if(type(value) == type(dicom.sequence.Sequence()))
        #  but I don't see the relevance of these tags anymore, so set them to NO

        if info == "dicom":
            dicomfields = [
                ["0008,0022", "Acquisition Date"],
                ["0008,0032", "Acquisition Time"],
                ["0008,0060", "Modality"],
                ["0008,0070", "Manufacturer"],
                ["0008,1010", "Station Name"],
                ["0008,103e", "Series Description"],
                ["0008,1010", "Station Name"],
                ["0018,0022", "Scan Options"], # Philips
                ["0018,0050", "Slice Thickness"],
                ["0018,0060", "kVp"],
                ["0018,0088", "Spacing Between Slices"], # Philips
                ["0018,0090", "Data Collection Diameter"],
                ["0018,1020", "Software Versions(s)"],
                ["0018,1030", "Protocol Name"],
                ["0018,1100", "Reconstruction Diameter"],
                ["0018,1120", "Gantry/Detector Tilt"],
                ["0018,1130", "Table Height"],
                ["0018,1140", "Rotation Direction"],
                ["0018,1143", "Scan Arc"], # noPhilips noSiemens
                ["0018,1150", "Exposure Time ms"], #Siemens
                ["0018,1151", "X-ray Tube Current"],
                ["0018,1152", "Exposure mAs"], # mA*tRot/pitch; tRot=exposure time
                ["0018,9345", "CTDIvol"],
                ["0018,1160", "Filter Type"],
                ["0018,1210", "Convolution Kernel"],
                ["0018,5100", "Patient Position"],
                ["0020,0013", "Image Number"],
                ["0020,1041", "Slice Location"],
                ["0028,0030", "Pixel Spacing"],
                ["01F1,1027", "Rotation Time"], # Philips
                ["01F1,104B", "Collimation"], # Philips
                ["01F1,104E", "Protocol"] ] # Philips

        elif info == "idose":
            dicomfields = [
                #"0018,9323", "Recon",
                ["01F7,109B", "iDose"] ]

        elif info == "id":
            dicomfields = [
                ["0018,1030", "ProtocolName"],
                ["0008,103e", "SeriesDescription"],
                ["0008,0022", "AcquisitionDate"],
                ["0008,0032", "AcquisitionTime"]
            ]

        results = []
        for df in dicomfields:
            key = df[0]
            value = ""
            if key=="0018,0088": #DICOM spacing
                value = self.dicomspacing_read()
            else:
                try:
                    value = self.dicomtag_read(key)
                except:
                    value = ""
            if key=="0018,1020":
                value = "'"+value
            results.append( (df[1],value) )

        return results

    def save_image_with_rois(self, fname):
        """
        save image with ROIs
        """
        # first the base image; take the raw image
        if self.dicomMode == wadwrapper_lib.stMode2D:
            im = self.pixeldataIn 
        else:
            im = self.pixeldataIn[self.unif_slice]

        save_image_with_rois(im, fname, circle_rois=self.results.gui_rois)

