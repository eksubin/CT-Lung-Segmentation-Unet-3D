import numpy as np
import SimpleITK as sitk
import os


# generating array from NRRD files
def generate_array_from_nrrd(path):
    img = sitk.ReadImage(path,sitk.sitkFloat32)
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = np.moveaxis(img_arr,[0],[2])
    img_arr = np.expand_dims(img_arr,  axis=-1)
    img_arr = np.expand_dims(img_arr, axis=0)
    print(img_arr.shape)
    return img_arr[:,:,4:]

# Saving network output as NRRD
def save_network_output_as_nrrd(input_volume_name, network_output,input_path, out_path, save_label=True):
    if save_label:
        input_volume_name = str(input_volume_name).replace('.nrrd','') + '.seg.nrrd'
    img_X = sitk.ReadImage(os.path.join(input_path,input_volume_name), sitk.sitkFloat32)
    origin =img_X.GetOrigin()
    direction = img_X.GetDirection()
    space = img_X.GetSpacing()
    savedImg = sitk.GetImageFromArray(np.squeeze(network_output))
    if save_label==False:
        savedImg = savedImg[:,:,4:]
    sitk.WriteImage(savedImg, os.path.join(out_path,input_volume_name))