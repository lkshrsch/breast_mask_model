#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 12:41:09 2022

@author: deeperthought
"""



import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import measure
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

GPU = 0
import tensorflow as tf
if tf.__version__[0] == '1':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list="0"
    tf.keras.backend.set_session(tf.Session(config=config))

elif tf.__version__[0] == '2':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only use the first GPU
      try:
        tf.config.experimental.set_visible_devices(gpus[GPU], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[GPU], True)
      except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

#%%

VISUALIZE = True

SCAN_PATH = '/Path/to/T1_scan.nii' # Can be sagittal or axial

MODEL_PATH = '/models/breast_mask_model.h5'

#%%

def dice_loss(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true * y_pred)
  denominator = tf.math.reduce_sum(y_true + y_pred)
  return 1 - numerator / denominator


def dice_coef_multilabel_bin0(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,0] * y_pred[:,:,:,0])
  denominator = tf.math.reduce_sum(y_true[:,:,:,0] + y_pred[:,:,:,0])
  return numerator / denominator

def dice_coef_multilabel_bin1(y_true, y_pred):
  numerator = 2 * tf.math.reduce_sum(y_true[:,:,:,1] * y_pred[:,:,:,1])
  denominator = tf.math.reduce_sum(y_true[:,:,:,1] + y_pred[:,:,:,1])
  return numerator / denominator



def get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch) :

    x_ = range(D1-(output_dpatch[0]//2),D1+((output_dpatch[0]//2)+output_dpatch[0]%2))
    y_ = range(D2-(output_dpatch[1]//2),D2+((output_dpatch[1]//2)+output_dpatch[1]%2))
    z_ = range(D3-(output_dpatch[2]//2),D3+((output_dpatch[2]//2)+output_dpatch[2]%2))
    
    x_norm = np.array(x_)/float(img_shape[0])  
    y_norm = np.array(y_)/float(img_shape[1])  
    z_norm = np.array(z_)/float(img_shape[2])  
    
    x, y, z = np.meshgrid(x_norm, y_norm, z_norm, indexing='ij')    
    coords = np.stack([x,y,z], axis=-1)
    return coords
   
def extractCoordinates(shapes, voxelCoordinates, output_dpatch):
    """ Given a list of voxel coordinates, it returns the absolute location coordinates for a given patch size (output 1x9x9) """

    all_coordinates = []
    for i in np.arange(len(shapes)):
        img_shape = shapes[i]
        for j in np.arange(len(voxelCoordinates[i])):     
            D1,D2,D3 = voxelCoordinates[i][j]
            all_coordinates.append(get_Coordinates_from_target_patch(img_shape,D1,D2,D3, output_dpatch))                    
    return np.array(all_coordinates)    




def get_breast_mask(SCAN_PATH, breastMask_model):

    X = nib.load(SCAN_PATH).get_data()
    X = X/float(np.percentile(X, 95))
    data = resize(X, output_shape=(X.shape[0],256,256), preserve_range=True, anti_aliasing=True, mode='reflect')   
    
    breaskMask = np.zeros(data.shape)
    for SLICE in range(1,data.shape[0]-1):   
        X_slice = data[SLICE-1:SLICE+2]
        voxelCoordinates = [[[SLICE,128,128]]]
        shapes = [data.shape]
        coords = extractCoordinates(shapes, voxelCoordinates, output_dpatch=[1,256,256])
        
        X_slice = np.expand_dims(X_slice,-1)
        X_slice = np.expand_dims(X_slice,0)
        
        y_pred = breastMask_model.predict([X_slice, coords])
        
        breaskMask[SLICE] = y_pred[0,0,:,:,1]
        
    return breaskMask
    
def keep_largest_component_and_fill_holes(volume, dilation_iters=3):
    """
    Keep only the largest connected component in a 3D binary volume,
    and fill internal holes using morphological closing (dilation then erosion).
    
    Parameters:
        volume (ndarray): 3D binary input volume.
        dilation_iters (int): Number of iterations for dilation and erosion.
    
    Returns:
        ndarray: Cleaned 3D binary volume.
    """
    # Label all connected components
    labeled = measure.label(volume, connectivity=3)
    props = measure.regionprops(labeled)
    
    if not props:
        return np.zeros_like(volume, dtype=bool)
    
    # Identify the largest component
    largest_label = props[np.argmax([p.area for p in props])].label
    largest_component = (labeled == largest_label)

    # Define 3D structuring element (26-connectivity)
    structure = generate_binary_structure(3, 2)

    # Morphological closing to fill holes: dilation followed by erosion
    closed = largest_component.copy()
    for _ in range(dilation_iters):
        closed = binary_dilation(closed, structure=structure)
    for _ in range(dilation_iters):
        closed = binary_erosion(closed, structure=structure)

    return closed

#%%

if __name__ == "__main__":
    
    my_custom_objects = {'Generalised_dice_coef_multilabel2':dice_loss,
                                     'dice_coef_multilabel_bin0':dice_coef_multilabel_bin0,
                                     'dice_coef_multilabel_bin1':dice_coef_multilabel_bin1}
    
    
    print(f'Loading model from: {MODEL_PATH}')
    breastMask_model = tf.keras.models.load_model(MODEL_PATH, custom_objects = my_custom_objects)
    
    print(f'Running inference on scan at: {SCAN_PATH}')
    ypred = get_breast_mask(SCAN_PATH, breastMask_model)
    
    mask = ypred > 0.5
    
    print('Post-processing mask..')
    mask = keep_largest_component_and_fill_holes(mask, dilation_iters=3)

    if VISUALIZE:
        img = nib.load(SCAN_PATH).get_fdata()
        img = resize(img, output_shape=(img.shape[0],256,256), preserve_range=True, anti_aliasing=True, mode='reflect')   
       
        SLICE = img.shape[0]//4
        
        plt.figure(figsize=(8,8))
        plt.subplot(221); plt.title('Input image')
        plt.imshow(img[SLICE], cmap='gray'); plt.xticks([]); plt.yticks([])

        plt.subplot(222); plt.title('Model prediction')
        plt.imshow(ypred[SLICE], cmap='gray'); plt.xticks([]); plt.yticks([])

        plt.subplot(223); plt.title('Resulting breast mask')
        plt.imshow(mask[SLICE], cmap='gray'); plt.xticks([]); plt.yticks([])


        plt.subplot(224); plt.title('Masked Input Image')
        
        plt.imshow(mask[SLICE]*img[SLICE], cmap='gray'); plt.xticks([]); plt.yticks([])
