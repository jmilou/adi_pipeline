# -*- coding: utf-8 -*-
#! /usr/bin/env python

"""
Module to inject a disk model in a cube
"""
__author__ = 'J. Milli @ ESO'
__all__ = ['createFakeDiscCube']

import numpy as np
import cv2

def createFakeDiscCube(discMap, parang, psf=None, cy=None, cx=None):
    """ Rotates an ADI cube to a common north given a vector with the 
    corresponding parallactic angles for each frame of the sequence. By default
    bicubic interpolation is used (opencv). 
    
    Parameters
    ----------
    discMap : array_like 
        Input image of a fake disc
    parang : list
        Vector containing the parallactic angles.
    cy, cx : float, optional
        Coordinates X,Y  of the point with respect to which the rotation will be 
        performed. By default the rotation is done with respect to the center 
        of the frames; central pixel if the frames have odd size.
        
    Returns
    -------
    cube_fake_disc : array_like
        Resulting cube with the fake disc inserted at the correct parractic angle
        
    """
    if not discMap.ndim == 2:
        raise TypeError('Input disc image is not a frame or 2d array.')
    disc = discMap.astype(np.float32)
    if psf is not None:
        if not psf.ndim == 2:
            raise TypeError('Input PSF is not a frame or 2d array.')
        if np.abs(np.sum(psf)-1) > 1e-3:
            print('Warning the PSF is not normalized to a total of 1.')
            print('Normalization was forced.')
            psf = psf/np.sum(psf)
        kernel = psf.astype(np.float32)
    if not parang.ndim == 1:
        raise TypeError('Input parallactic angle is not a 1d array')
    nframes=len(parang)
    ny, nx = disc.shape
    cube_fake_disc = np.ndarray((nframes,ny,nx)) 
    cube_fake_disc.fill(0)
    if not cy and not cx:  cy, cx = frame_center(disc)
    
    for i in xrange(nframes): 
        M = cv2.getRotationMatrix2D((cx,cy), -parang[i], 1)
        if psf is None:
            cube_fake_disc[i] = cv2.warpAffine(disc, M, (nx, ny))
        else:
            cube_fake_disc[i] = cv2.filter2D(cv2.warpAffine(\
                discMap.astype(np.float32), M, (nx, ny)),-1,kernel)             
#            cube_fake_disc[i] = convolve2d(cv2.warpAffine(\
#                discMap.astype(np.float32), M, (nx, ny)),psf,mode='same')             
    return cube_fake_disc

def frame_center(array, verbose=False):
    """ Returns the coordinates y,x of a frame central pixel if the sides are 
    odd numbers. Python uses 0-based indexing, so the coordinates of the central
    pixel of a 5x5 pixels frame are (2,2). Those are as well the coordinates of
    the center of that pixel (sub-pixel center of the frame).
    """   
    y = array.shape[0]/2.       
    x = array.shape[1]/2.
    
    # If frame size is even
    if np.mod(array.shape[0],2)==0:
        cy = np.ceil(y)
    else:
        cy = np.ceil(y) - 1    # side length/2 - 1, python has 0-based indexing
    if np.mod(array.shape[1],2)==0:
        cx = np.ceil(x)
    else:
        cx = np.ceil(x) - 1

    cy = int(cy); cx = int(cx)
    if verbose:
        print('Center px coordinates at x,y = ({0:d},{1:d})'.format(cy, cx))
    return cy, cx

    