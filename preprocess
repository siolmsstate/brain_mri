# Sample codes for preprocessing the data
import os
import numpy as np
import pydicom as dicom
from rt_utils import RTStructBuilder
import re

# Save MRI dicom as numpy
PathDicom = r"Dataset/1950-01__Studies"
patient_path = "MRI Brain and Treatment"
course_id = "1"
for foldername in os.listdir(PathDicom):
    if '_MR_' in foldername:
        listFilesDCM = []
        os.mkdir(os.path.join(patient_path, str(foldername[0:7] + course_id)))
        for filename in os.listdir(os.path.join(PathDicom, foldername)):
            if ".dcm" in filename.lower():
                listFilesDCM.append(os.path.join(PathDicom, foldername, filename))
                RefDs = dicom.read_file(listFilesDCM[0])
                ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(listFilesDCM))
                ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
                for filenameDCM in listFilesDCM:
                    ds = dicom.read_file(filenameDCM)
                    ArrayDicom[:, :, listFilesDCM.index(filenameDCM)] = ds.pixel_array
                np.save(os.path.join(patient_path, str(foldername[0:7] + course_id)) + '/'
                        + str(foldername[0:7]) + course_id + '_MR_t1.npy', ArrayDicom)

# Save RT structure dicom as numpy mask (Similar to RT dose)
for rtfoldername in os.listdir(PathDicom):
    if '_RTst_' in rtfoldername:
        rt_struct_path = os.path.join(PathDicom, rtfoldername)
        patient_id = rtfoldername[0:6]
        for foldername in os.listdir(PathDicom):
            if '_MR_' in foldername and str(patient_id) in foldername:
                dicom_series_path = os.path.join(PathDicom, foldername)
                for rtfilename in os.listdir(rt_struct_path):
                    if ".dcm" in rtfilename.lower():
                        rt_struct_path = os.path.join(rt_struct_path, rtfilename)
                        rtstruct = RTStructBuilder.create_from(
                            dicom_series_path=dicom_series_path,
                            rt_struct_path=rt_struct_path
                        )
                        names = rtstruct.get_roi_names()
                        for name in names:
                            if "Skull" not in name: # exclude skull annotations
                                mask_3d = rtstruct.get_roi_mask_by_name(name)
                                mask = mask_3d * 1
                                np.save(os.path.join(patient_path, str(foldername[0:7] + course_id)) + '/'
                                + str(foldername[0:7])+ course_id + '_L_' + re.sub('\W+','', str(name)) + '.npy', mask)

# Get training samples
def crop_les(d):
    true_points = np.argwhere(d)
    top_left = true_points.min(axis=0)
    bottom_right = true_points.max(axis=0)
    cropped_arr = d[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1, top_left[2]:bottom_right[2]+1]
    return cropped_arr

def re_arr(arrpath):
    global mri_path, mask_path
    for npyfile in os.listdir(arrpath):
        if "_MR_" in npyfile:
            mri_path = os.path.join(folderpath, npyfile)
        else:
            mask_path = os.path.join(folderpath, npyfile)
    return mri_path, mask_path

sample_path = "..." # input path
output_path = "..." # output_path
for foldername in os.listdir(sample_path):
    folderpath = os.path.join(sample_path, foldername)
    patient_id = foldername[0:6]
    mri_path, mask_path = re_arr(folderpath)
    mri_arr = np.load(mri_path)
    les_mask_arr = np.load(mask_path)
    les_arr = les_mask_arr*mri_arr
    c_les_arr = crop_les(les_arr)
    print(c_les_arr.shape)
    np.save(output_path, les_arr)
