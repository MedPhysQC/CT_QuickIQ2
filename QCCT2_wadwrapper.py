#!/usr/bin/env python
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
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://bitbucket.org/MedPhysNL/wadqc/wiki/Home
# 
#
# Changelog:
#   20230906: remove deprecated warning; Pillow 10.0.0
#   20210422: first new config version
#   20210413: new version based on Philips_QuickIQ version 20201002; changed name of output image; 
#             added support for updated philips phantoms; added testing scripts
# ./QCCT2_wadwrapper.py -c Config/ct2_philips_umcu_series_mx8000idt.json -d TestSet/StudyMx8000IDT -r results_mx8000idt.json

__version__ = '20230906'
__author__ = 'aschilham'
USE_GUI = True
USE_GUI = False

import os
# this will fail unless wad_qc is already installed
from wad_qc.module import pyWADinput
from wad_qc.modulelibs import wadwrapper_lib

if not 'MPLCONFIGDIR' in os.environ:
    try:
        # new method
        from importlib.metadata import version as pkg_version
    except:
        # deprecated method
        import pkg_resources
        def pkg_version(what):
            return pkg_resources.get_distribution(what).version
    try:
        #only for matplotlib < 3 should we use the tmp work around, but it should be applied before importing matplotlib
        matplotlib_version = [int(v) for v in pkg_version("matplotlib").split('.')]
        if matplotlib_version[0]<3:
            os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 
    except:
        os.environ['MPLCONFIGDIR'] = "/tmp/.matplotlib" # if this folder already exists it must be accessible by the owner of WAD_Processor 

if not USE_GUI:
    import matplotlib
    matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.

try:
    import pydicom as dicom
except ImportError:
    import dicom
import QCCT2_lib

def logTag():
    return "[QCCT_wadwrapper] "

# helper functions
def override_settings(cs, params):
    """
    Look for 'use_' params in to force behaviour of module
    """
    # verbosity
    try:
        cs.verbose = params['verbose']
    except:
        pass


##### Real functions
def qc_series(data, results, action, override={}):
    """
    QCCT_UMCU Checks: extension of Philips QuickIQ (also for older scanners without that option), for both Head and Body if provided
      Uniformity
      HU values
      Noise
      Linearity 

    Workflow:
        1. Read image or sequence
        2. Run test
        3. Build xml output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    # overrides from test scripts
    for k,v in override.items():
        params[k] = v

    dcmInfile, pixeldataIn, dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], headers_only=False, 
                                                                    logTag=logTag(), do_transpose=False)
    qclib = QCCT2_lib.CT_QC(dcmInfile, pixeldataIn, dicomMode)
    qclib.verbose = False
    qclib.verbose = USE_GUI
    override_settings(qclib, params)

    ## 2. Run tests
    # QC
    error, msg = qclib.run(params)
    if error:
        raise ValueError("{} ERROR! {}".format(logTag, msg))

    cs = qclib.results
    if qclib.verbose and qclib.hasmadeplots:
        import matplotlib.pyplot as plt
        plt.show()

    ## Struct cs now contains all the results and we can write these to the WAD IQ database
    
    # summary values
    hei,wid = qclib.work_im.shape
    res = [
        ('MeanCenter', cs.unif_roiavg[0]),
        ('MeanAir', cs.lin_roiavg[1]),
        ('shiftxpx', cs.phantom_xcycdiampx[0]-(wid-1)/2.),
        ('shiftypx', cs.phantom_xcycdiampx[1]-(hei-1)/2.),

        # fix names: start; backwards compatible names
        ('unif', cs.uniformity),
        ('roisd', cs.noise_roistd[0]),
        ('snr_hol', cs.noise_roisnr[0]),
        ('linearity', cs.linearity_fit),
        ('maxdev', cs.linearity_dev),
        # fix names: end

        # new values
        ('diampx', cs.phantom_xcycdiampx[2]),
    ]
    for key,val in res:
        results.addFloat(key, val)

    # new values
    for i,(avg,std) in enumerate(zip(cs.unif_roiavg, cs.unif_roistd)):
        results.addFloat("unif_avg_{}".format(i), avg)
        results.addFloat("unif_std_{}".format(i), std)
    for i,(avg,std,gt) in enumerate(zip(cs.lin_roiavg, cs.lin_roistd, cs.lin_roiGT)):
        results.addFloat("lin_avg_{}".format(i), avg)
        results.addFloat("lin_std_{}".format(i), std)
        #results.addFloat("lin_gt_{}".format(i), gt)
 
    ## Build thumbnail
    prefix = results._out_path.split(".json")[0]
    filename = '{}.jpg'.format(prefix) # Use jpg if a thumbnail is desired
    qclib.save_image_with_rois(filename)
    varname = 'CTslice'
    results.addObject(varname, filename) 
    if qclib.verbose and qclib.hasmadeplots:
        import matplotlib.pyplot as plt
        plt.show()

def acqdatetime_series(data, results, action):
    """
    Read acqdatetime from dicomheaders and write to IQC database

    Workflow:
        1. Read only headers
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    ## 1. read only headers
    dcmInfile = dicom.read_file(data.series_filelist[0][0], stop_before_pixels=True)

    dt = wadwrapper_lib.acqdatetime_series(dcmInfile)

    results.addDateTime('AcquisitionDateTime', dt) 

def header_series(data, results, action):
    """
    Read selected dicomfields and write to IQC database

    Workflow:
        1. Run tests
        2. Build xml output
    """
    try:
        params = action['params']
    except KeyError:
        params = {}

    info = 'dicom'
    dcmInfile, pixeldataIn, dicomMode = wadwrapper_lib.prepareInput(data.series_filelist[0], headers_only=True, 
                                                                    logTag=logTag(), do_transpose=False)
    qclib = QCCT2_lib.CT_QC(dcmInfile, pixeldataIn, dicomMode)
    qclib.verbose = False
    override_settings(qclib, params)

    ## 1. Run tests
    dicominfo = qclib.dicom_info(qclib, info)

    ## 2. Add results to 'result' object
    floatlist = [
        "Slice Thickness",
        "kVp",
        "Spacing Between Slices", # Philips
        "Data Collection Diameter",
        "Reconstruction Diameter",
        "Gantry/Detector Tilt",
        "Table Height",
        "Scan Arc", # noPhilips noSiemens
        "Exposure Time ms", #Siemens
        "X-ray Tube Current",
        "Exposure mAs", # mA*tRot/pitch; tRot=exposure time
        "CTDIvol",
        "Image Number",
        "Slice Location",
        "Rotation Time", # Philips
    ]    

    for di in dicominfo:
        varname = di[0]
        if varname in floatlist:
            # store these values as floats
            try:
                results.addFloat(varname, float(di[1]))
            except ValueError:
                results.addString(varname, str(di[1])[:min(len(str(di[1])),100)])
        else:
            results.addString(varname, str(di[1])[:min(len(str(di[1])),100)])

def main(override={}):
    """
    override from testting scripts
    """
    data, results, config = pyWADinput()

    # plugionversion is newly added in for this plugin since pywad2
    varname = 'pluginversion'
    results.addString(varname, str(QCCT2_lib.__version__))

    # read runtime parameters for module
    for name,action in config['actions'].items():
        if name == 'acqdatetime':
            acqdatetime_series(data, results, action)

        elif name == 'header_series':
            header_series(data, results, action)
        
        elif name == 'qc_series':
            qc_series(data, results, action, override)

    #results.limits["minlowhighmax"]["mydynamicresult"] = [1,2,3,4]

    results.write()
    
if __name__ == "__main__":
    # main in separate function to be called by ct_tester
    main()
    