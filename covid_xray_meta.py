
# import planck_meta_regen

import mscale_covid_xray_prior2

# def covid_xray_meta():

write_freq = 8
init_roi = 3
limit_roi = 26
roi_incr = 1

mean_min_ref = 0
rescaling = 1
mad_vpk2pk_scaling = 1

sm_fac = 0.5
sm_fac_order = 0
sm_fac_l2l = 0.001
sm_fac_l2l_order = 0

log_sw = 1
log_off = 1.01

img_name = 'covid_action_ex12b'
ext = '.jfif'

rgb_idx = 1   # green
img_dbg_inter = 0
img_trim_fac = 0.2

mscale_covid_xray_prior2.uk_xray('./', img_name, ext, sm_fac, sm_fac_order, sm_fac_l2l, sm_fac_l2l_order, 'covid_chexray_minref_rescaling_',
                                     init_roi, limit_roi, roi_incr, 0.0000001, write_freq, log_sw, log_off, 1, 1, rescaling, 0, mean_min_ref, mad_vpk2pk_scaling, rgb_idx, img_dbg_inter, img_trim_fac)


img_name = 'kaggle_normal_ex12'
ext = '.jpeg'

mscale_covid_xray_prior2.uk_xray('./', img_name, ext, sm_fac, sm_fac_order, sm_fac_l2l, sm_fac_l2l_order, 'covid_chexray_minref_rescaling_',
                                     init_roi, limit_roi, roi_incr, 0.0000001, write_freq, log_sw, log_off, 1, 1, rescaling, 0, mean_min_ref, mad_vpk2pk_scaling, rgb_idx, img_dbg_inter, img_trim_fac)

