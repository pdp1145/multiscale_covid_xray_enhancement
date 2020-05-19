
# import planck_meta_regen

import mscale_covid_xray_prior

write_freq = 1
init_roi = 3
limit_roi = 25
roi_incr = 1

sm_fac = 0.001
sm_fac_order = 0
sm_fac_l2l = 0.001
sm_fac_l2l_order = 0

log_sw = 1
log_off = 1.01

# img_name = 'case41_Xray_1'
# img_name = 'eunn1'
# ext = '.jfif'
img_name = 'eury_covid'
ext = '.jfif'
rgb_idx = 1;   # green
img_dbg_inter = 0;
img_trim_fac = 0.2

mscale_covid_xray_prior.uk_xray('./', img_name, ext, sm_fac, sm_fac_order, sm_fac_l2l, sm_fac_l2l_order, 'covid_chexray_madmean_roi_',
                                     init_roi, limit_roi, roi_incr, 0.0000001, write_freq, log_sw, log_off, 1, 1, 0, 0, 0, 0, rgb_idx, img_dbg_inter, img_trim_fac)
