
# import planck_meta_regen

import mscale_covid_xray_prior_dirconts

import os, fnmatch

# def proc_covid_dir_conts(dir_name):

def covid_xray_meta_wfname_spec(dir_name, img_name_f, res_dir):

    img_name, img_ext = os.path.splitext(img_name_f)

    write_freq = 8
    init_roi = 3
    limit_roi = 11
    roi_incr = 1

    mean_min_ref = 0
    rescaling = 0

    sm_fac = 0.001
    sm_fac_order = 0
    sm_fac_l2l = 0.001
    sm_fac_l2l_order = 0

    log_sw = 1
    log_off = 1.01

    rgb_idx = 1   # green
    img_dbg_inter = 0
    img_trim_fac = 0.2
    mscale_covid_xray_prior_dirconts.uk_xray(dir_name, img_name, img_ext, res_dir, sm_fac, sm_fac_order, sm_fac_l2l, sm_fac_l2l_order, 'covid_chexray_minref_logscaled_',
                                         init_roi, limit_roi, roi_incr, 0.0000001, write_freq, log_sw, log_off, 1, 1, 0, rescaling, mean_min_ref, 0, rgb_idx, img_dbg_inter, img_trim_fac)

dir_name = "/home/pdp1145/chest_xray_imgs/Covid_action/Source1/"
# dir_name = "/home/pdp1145/chest_xray_imgs/chest_xray/chest_xray_kaggle/train/NORMAL/"
# dir_name = "/home/pdp1145/chest_xray_imgs/chest_xray/chest_xray_kaggle/train/PNEUMONIA/"

listOfFiles = os.listdir(dir_name)
pattern = "*.jfif"
for entry in listOfFiles:
    if fnmatch.fnmatch(entry, pattern):
        print (entry)
        covid_xray_meta_wfname_spec(dir_name, entry, '/home/pdp1145/multiscale_covid_xray_enhancement_repo/')
