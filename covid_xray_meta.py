
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

# #
# # img_name = 'eso1907a'
# #
# img_name = 'm87'
# img_name = 'case41_Xray_1'
img_name = 'eunn1'
ext = '.jfif'
rgb_idx = 1;   # green
img_dbg_inter = 0;
mscale_covid_xray_prior.uk_xray('./', img_name, ext, sm_fac, sm_fac_order, sm_fac_l2l, sm_fac_l2l_order, 'covid_chexray_madmean_roi_',
                                     init_roi, limit_roi, roi_incr, 0.0000001, write_freq, 0, 1.01, 1, 1, 0, 0, 0, 0, rgb_idx, img_dbg_inter)
