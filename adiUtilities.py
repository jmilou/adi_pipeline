# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 12:12:01 2015

@author: jmilli
"""
#import sys
#sys.path.append('/Users/jmilli/Dropbox/lib_py/image_utilities') # add path to our file
# we no longer use the package rot because of the opencv issue
#import rotation_images as rot
import scipy.ndimage.interpolation
import numpy as np
from scipy.signal import convolve2d
import radial_data as rd
from scipy.stats import trim_mean
#import pyfits as pf
#from sklearn.decomposition import PCA

def collapseCube(cube,trim=0.5,mask=None):
    """
    Collapse a cube using the median or a trimmed mean.
    Input:
        - cube: a cube of images
        - trim: The collaspe along the cube 3rd dimension is a trimmed mean. It expects
                     a value between 0 (mean) and 0.5 (median) corresponding to the 
                     fraction of frames to remove on both sides (lower side and upper side) before 
                     computing the mean of the cube along the axis 0. For 
                     trim=0.1, this removes 10% of the lowest and highest pixels
                     along the third axis of the cube before collapsing the cube
        - mask: a 3d binary array with True for values to be discarded when collapsing
                the cube
    Output:
        - collapsed image
    """
    if mask != None:
        cube = np.copy(cube)
        print('Applying a binary mask: True values are discarded')
        cube[mask] = np.nan
    if trim==0.5:
        image = np.nanmedian(cube,axis=0)
    elif trim==0.:
        image =np.nanmean(cube,axis=0)
    elif trim>0. and trim < 0.5:
        image = trim_mean(cube, trim, axis=0)
    else:
        raise ValueError('The clean mean expects value between 0. (mean) \
        and 0.5 (median), received {0:4.2f}'.format(trim))
    return image

def subtractMedian(cube,trim=0.5,mask=None):
    """
    Subtracts the median or clean mean of a cube (first step of ADI)
    Input:
        - cube: a cube of images
        - trim: if 0.5 (default value), it subtracts the median. Otherwise expects
                     a value between 0 (mean) and 0.5 (median) corresponding to the 
                     fraction of frames to remove on both sides (lower side and upper side) before 
                     computing the mean of the cube along the axis 0. For 
                     trim=0.1, this removes 10% of the lowest and highest pixels
                     along the third axis of the cube before collapsing the cube
        - mask: a 3d binary array with True for bad values to be replaced by nan
                before computing the star signal
    Output:
        - residual cube after subtraction
    """
    return cube - collapseCube(cube,trim=trim,mask=mask)

def derotateCollapse(cube,parang,rotoff=0.,trim=0.5,inplace=False):
    """
    Derotates a cube of images according ot the parallactic angle (parang) 
    and collapses it using the median or clean mean of a cube (first step of ADI).
    It uses scipy.ndimage.interpolation.rotate for derotation and therefore
    expects a cube with an odd number of rows and columns, centered on the central 
    pixel.
    Input:
        - cube: a cube of images
        - trim: The collaspe along the cube 3rd dimension is a trimmed mean. It expects
                     a value between 0 (mean) and 0.5 (median) corresponding to the 
                     fraction of frames to remove on both sides (lower side and upper side) before 
                     computing the mean of the cube along the axis 0. For 
                     trim=0.1, this removes 10% of the lowest and highest pixels
                     along the third axis of the cube before collapsing the cube
    Output:
        - collapsed image after derotation
    """
    nbFrames=cube.shape[0]
    if len(parang) != nbFrames:
        raise IndexError('The cube has {0:5d} frames and the parallactic angle array only {1:5d} elements'.format(nbFrames,len(parang)))
    if inplace:
        for i in range(nbFrames):
            cube[i,:,:] = scipy.ndimage.interpolation.rotate(\
                cube[i,:,:],-(parang[i]-rotoff),reshape=False,order=1,prefilter=False)
        return collapseCube(cube,trim=trim)
    else:            
        cubeDerotated = np.ndarray(cube.shape)
        for i in range(nbFrames):
    #        cubeDerotated[i,:,:] = rot.frame_rotate(cube[i,:,:],-parang[i]+rotoff)
            cubeDerotated[i,:,:] = scipy.ndimage.interpolation.rotate(\
                cube[i,:,:],-(parang[i]-rotoff),reshape=False,order=1,prefilter=False)
        return collapseCube(cubeDerotated,trim=trim)

def insertFakeDisc(cube,parang,discMap,rotoff=0.,psf=None,copy=True):
    """
    Inserts a disc image in a cube according to the parallactic angle
    Input:
        - cube: a cube of images
        - parang: a list of parallactic angle of the same length as cube.shape[0]
        - discMap: an image of size cube.shape[1] x cube.shape[2] of the disc (unconvolved)
        - rotoff: the angular offset to be applied before the insertion (0 by default)
        - psf: if None, no convolution is done, if an image is provided, then a convolution
              by this image is done. The center of the psf must be in the pixel
              nx/2-1,nx/2-1 (to be checked again !)
        -copy: boolean. If True (default), does not modify the argument cube but
            creates a copy of the cube. If False, modifies the cube.
    """
    if copy:
        cubeInserted = np.copy(cube)
    else:
        cubeInserted = cube
    if not cube.ndim == 3:
        raise TypeError('Input array is not a cube or 3d array.')    
    nbFrames,ny,nx = cube.shape
    if nx % 2 == 0 or ny % 2 == 0:
        print(nx % 2)
        print(nx % 2 == 0)
        print(ny % 2)
        print(ny % 2 ==0)
        print(nx % 2 == 0 or ny % 2 == 0)
        raise IndexError('The frames do not have an odd row or column number: {0:5d}px x {1:5d}px'.format(nx,ny)) 
    if len(parang) != nbFrames:
        raise IndexError(\
        'The cube has {0:5d} frames and the parallactic angle array only {1:5d} elements'.format(nbFrames,len(parang)))
    for i in range(nbFrames):
#        discMapRotated = rot.frame_rotate(discMap,parang[i]-rotoff)
        discMapRotated = scipy.ndimage.interpolation.rotate(discMap,\
            parang[i]-rotoff,reshape=False,order=1,prefilter=False)
        if psf is not None:
            discMapRotated=convolve2d(discMapRotated,psf,mode='same')           
        cubeInserted[i,:,:] += discMapRotated
    return cubeInserted

def flatten_image(image):
    """
    Subtract the azimuthal median from an image
    """
    image_rd = rd.Radial_data(image)
    return image-image_rd.azimuthalmedianmap

def flatten_cube(cube):
    """
    Subtract the azimuthal median from each image of a cube
    """
    nb=cube.shape[0]
    cube_flattened = np.zeros_like(cube)
    for i in range(nb):
        cube_flattened[i,:,:] = flatten_image(cube[i,:,:])
    return cube_flattened

def simple_reference_subtraction(science_cube,reference_cube,mask):
    """
    Given a science cube, a reference cube and a mask, the function finds 
    the best scaling to apply to the reference frames to minimize the residuals
    (norm 2) and subtracts the scaled frames from the science cube.  
    """
    nframes = science_cube.shape[0]
    if reference_cube.shape[0] != nframes:
        print('The number of references is different from the number of science frames. Returning')
        return
    subtracted_cube = np.ndarray(science_cube.shape)
    for i in range(nframes):
        tmp = reference_cube[i,:,:]*mask
        coeff = np.sum(science_cube[i,:,:]*tmp)/np.sum(tmp**2)
        subtracted_cube[i,:,:] = science_cube[i,:,:]-coeff*reference_cube[i,:,:]
    return subtracted_cube


#path='/Volumes/jmilli/Documents/HR4796_irdis/20141209_reduction/flux_study/python_input/'
#path='/Volumes/MILOU_1TB/HR4796_CI/python_input/'
#from astropy.io import fits
#import os
#
#cubeHDU = fits.open(os.path.join(path,'outCube_skycorrected_total.fits'))
#cube=cubeHDU[0].data
#cubeHDU.close()
#
#nframes,ny,nx = cube.shape
#print('The frames size is: {0}px x {1}px and the center is [{2},{3}]'.format(nx,ny,nx/2,ny/2)) 
#cube = cube[:,1:,1:]
#nframes,ny,nx = cube.shape
#print('The frames size is now: {0}px x {1}px and the center is [{2},{3}]'.format(nx,ny,nx/2,ny/2)) 
#
#frame = 'outIm_restmean_mcADI_total.fits'
#hdulist = fits.open(os.path.join(path, frame))
#img = hdulist[0].data
#hdulist.close()
#
#img = img[1:,1:]
#
#mask_nan = np.isnan(img)
#ny,nx=img.shape
#
#region_name = os.path.join(path,'chi2_minimization_region_ADI.reg')
#import pyregion
#with open (region_name, "r") as myfile:
#    ellipse_region_string=myfile.read() #.replace('\n', '')
#ellipse_region = pyregion.parse(ellipse_region_string)
#
#import pyds9
#ds9=pyds9.DS9()
#ds9.set_np2arr(img)
#ds9.set('regions',ellipse_region_string)
#
#import pidly
#idl = pidly.IDL()
#idl("!PATH = EXPAND_PATH('+/Users/jmilli/Dropbox/lib_idl/IPAG-ADI-v1.6.7/pro:/Users/jmilli/Dropbox/lib_idl/pro_disques:/Users/jmilli/Dropbox/lib_idl/PCA:/Users/jmilli/Dropbox/lib_i\
#dl/nnls:/Users/jmilli/Dropbox/lib_idl/pro_field_rotation_david:/Users/jmilli/Dropbox/lib_idl/some_onera_routines:/Users/jmilli/Dropbox/lib_idl/NICI_routines:/Users/jmilli/Drop\
#box/lib_idl/CAOS/CAOS_Library_5.2:/Users/jmilli/Dropbox/lib_idl/Keck:/Users/jmilli/Dropbox/lib_idl/zim_disk:/Users/jmilli/Dropbox/lib_idl/statisticsOB:+/Users/jmilli/Dropbox/l\
#ib_idl/coyote:/Users/jmilli/Dropbox/lib_idl/IRDIS_tools:/Users/jmilli/Dropbox/lib_idl/coyote:/Users/jmilli/Dropbox/lib_idl/buie_idlv71:/Users/jmilli/Dropbox/lib_idl/added_rout\
#ines:/Users/jmilli/Dropbox/lib_idl/sparta_extract_pro')+':' + !PATH")
#nx=50
#PA = 90
#itilt=0 
#e=0.0
#theta0=0
#dstar=5
#r0=10
#map_disc = idl.func('call_scimage_hr4796_python',PA=PA,flux_max=1.,
#                itilt=itilt,nx=nx,aniso_g=0.,xdo=0.,ydo=0.,r0=r0,
#                pfov=0.01225,ain=21.,aout=-15.,gamma=2.,ksi0=1.,
#                e=e,theta0=theta0)
#ds9.set_np2arr(map_disc)
#
#map_disc_t = idl.func('call_scimage_hr4796_python',PA=PA,flux_max=1.,
#                itilt=itilt,nx=nx-1,aniso_g=0.,xdo=0.,ydo=0.,r0=r0,
#                pfov=0.01225,ain=21.,aout=-15.,gamma=2.,ksi0=1.,
#                e=e,theta0=theta0)
#ds9.set_np2arr(map_disc_t)
#
#
#parangleHDU=fits.open(os.path.join(path,'parang.fits'))
#parangle=parangleHDU[0].data
#parangleHDU.close()
#
#rotoff=-134.87
#
#cubeSubtracted = subtractMedian(cube)
#reducedImage = derotateCollapse(cubeSubtracted,parangle,rotoff=rotoff)
#
