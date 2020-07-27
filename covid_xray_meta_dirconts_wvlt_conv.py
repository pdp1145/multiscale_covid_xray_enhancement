
# import planck_meta_regen

import mscale_covid_xray_wvlt_conv

import os, fnmatch

# def proc_covid_dir_conts(dir_name):

def covid_xray_meta_wfname_spec_l(dir_name, img_name_f, res_dir):

    img_name, img_ext = os.path.splitext(img_name_f)

    browser_disp = 0    # Use Firefox or not for high resolution (in intermediate results) set of displays
    write_freq = 5
    init_roi = 3
    limit_roi = 42
    roi_incr = 1

    mean_min_ref = 0
    rescaling = 0
    mad_vpk2pk_scaling = 1

    sm_fac = 0.05
    sm_fac_order = 0
    sm_fac_l2l = 0.001
    sm_fac_l2l_order = 0

    log_sw = 1
    log_off = 1.01

    rgb_idx = 1   # green
    img_dbg_inter = 0
    img_trim_fac = 0.2
    mscale_covid_xray_wvlt_conv.uk_xray(dir_name, img_name, img_ext, res_dir, sm_fac, sm_fac_order, sm_fac_l2l, sm_fac_l2l_order, 'covid_chexray_minref_nonrescaled_mad_vpk2pk_',
                                         init_roi, limit_roi, roi_incr, 0.0000001, write_freq, log_sw, log_off, 1, 1, rescaling, 0, mean_min_ref, mad_vpk2pk_scaling, rgb_idx, img_dbg_inter, img_trim_fac, browser_disp)


# dir_name = "/home/pdp1145/chest_xray_imgs/Covid_action/Source1/"
# pattern = "*.jfif"

# dir_name = "/home/pdp1145/chest_xray_imgs/chest_xray/chest_xray_kaggle/train/NORMAL/"
# pattern = "*.jpeg"

# dir_name = "/home/pdp1145/chest_xray_imgs/chest_xray/chest_xray_kaggle/train/PNEUMONIA/"
# pattern = "*.jpeg"

# dir_name = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/non_covid19_imgs/"
# pattern = "*.jpg"

# dir_name = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/normal/"
# dir_name = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/covid19/"
# pattern = "*.jpg"

# High res images for wavelet based / convolution / local variance weighting dev:
#
dir_name = "/home/pdp1145/chest_xray_imgs/montreal_dataset/covid-chestxray-dataset-master/covid19_exs/"
pattern = "*.png"

# dir_name = "/home/pdp1145/chest_xray_imgs/Stanford_valid_samples/"
# pattern = "*.jpg"

# dir_name = "/home/pdp1145/chest_xray_imgs/covid_chestxray_dataset_samples/"
# pattern = "*.jpeg"

# res_dir = '/home/pdp1145/multiscale_covid_xray_enhancement_repo2/'
# listOfFiles = os.listdir(dir_name)

# res_dir = '/home/pdp1145/multiscale_covid_xray_enhancement_repo_stanford/non_covid19/'
# listOfFiles = os.listdir(dir_name)

# res_dir = '/home/pdp1145/multiscale_covid_xray_enhancement_repo_stanford/normal/'
res_dir = '/home/pdp1145/multiscale_covid_xray_enhancement_repo_stanford/covid19/'
listOfFiles = os.listdir(dir_name)

for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):                                                     
        print (entry)
        # entry = 'EUNQ5iSWkAUfemx.jfif'
        covid_xray_meta_wfname_spec_l(dir_name, entry, res_dir)
        arf = 12
