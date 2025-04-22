# breast_mask_model
 2D U-Net for prediction of a breast mask on sagittal T1-weighted MRI


**Usage**

Script 'Get_breast_mask' only requires the following user inputs:

VISUALIZE = True  # to plot results

SCAN_PATH = '/Path/to/T1_scan.nii' # Input mri has to be in nifty format. Can be sagittal or axial.

MODEL_PATH = '/models/breast_mask_model.h5'

![Demo result](/figures/demo_result.png)
