import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl
import time
import nibabel
import trimesh
import Resample
from linetimer import CodeTimer
import pyvista
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
from nibabel import processing, nifti1, affines

# GPU BACKEND
# Resample.InitCUDA('Quadro GP100')
# ResampleFilterCOMPUTING_BACKEND = 'CUDA'
Resample.InitOpenCL('gfx1030')
ResampleFilterCOMPUTING_BACKEND = 'OpenCL'
# ResampleFilterCOMPUTING_BACKEND = 'CPU'

# LOAD FILES
print("Loading Files")
T1Conformal_nii = "d:/Shirshak/Research/BabelBrain/Tests/4000/LIFU_4000_after_holder-isotropic.nii.gz"
spatial_steps = {
    'low_res': 0.919, # 6 PPW, 200 kHz
    'medium_res': 0.340,
    'high_res': 0.223, # 6 PPW, 825 kHz
}
spatial_step = spatial_steps['high_res']
T1Conformal=nibabel.load(T1Conformal_nii)

# Calculate new affine
T1W_affine_upscaled = affines.rescale_affine(T1Conformal.affine.copy(),T1Conformal.shape,spatial_step,(int(T1Conformal.shape[0]//spatial_step)+1,int(T1Conformal.shape[1]//spatial_step)+1,int(T1Conformal.shape[2]//spatial_step)+1))

# output dimensions
T1W_data = T1Conformal.get_fdata()
T1W_nifti = nifti1.Nifti1Image(T1W_data,T1Conformal.affine)
output_data = np.zeros((int(T1W_data.shape[0]//spatial_step)+1,int(T1W_data.shape[1]//spatial_step)+1,int(T1W_data.shape[2]//spatial_step)+1))
output_nifti = nifti1.Nifti1Image(output_data,T1W_affine_upscaled)

# Run resample step
print(f"Starting Resample")
with CodeTimer("Finished resample step", unit="s"):
    T1W_resampled = Resample.ResampleFromTo(T1W_nifti,output_nifti,mode='constant',order=0,cval=T1W_data.min(),GPUBackend=ResampleFilterCOMPUTING_BACKEND)
    # T1W_resampled_truth = processing.resample_from_to(T1W_nifti,output_nifti,mode='constant',order=0,cval=T1W_data.min()) # Truth method

# PLOT
print(f"Starting Plot")
datas = [T1W_data,T1W_resampled.get_fdata()]
titles = ['Data', 'Resampled Data']
color_map = 'gray'
data_num = len(datas)
axes_num = 3

# Create plots
plt.figure()
for num in range(data_num):
    for axis in range(axes_num):
        plot_idx = axis * data_num + num + 1
        midpoint = datas[num].shape[axis]//2
        plt.subplot(axes_num,data_num,plot_idx)

        if axis == 0:
            plt.imshow(datas[num][midpoint,:,:], cmap=color_map)
        elif axis == 1:
            plt.imshow(datas[num][:,midpoint,:], cmap=color_map)
        else:
            plt.imshow(datas[num][:,:,midpoint], cmap=color_map)

        if titles is not None and axis == 0:
            plt.title(titles[num])
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5)
plt.show()
print("Done plotting")

# diffs = abs(opencl_data-cupy_data)
# plt.hist(diffs.flatten(), bins=[0,1,100,1000,diffs.max()], edgecolor='black')  # Adjust the number of bins as needed
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.title('Histogram')
# plt.grid(True)
# plt.show()
# print("END")