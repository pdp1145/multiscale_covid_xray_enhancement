
#
# 'mscale_covid_xray_prior':
#     > significant cleanup
#     > restructuring combination of layers over scale using file storage for more precise weighting of each layer
#
#


# from bokeh.plotting import figure, show
import matplotlib.pyplot as plt
# from bokeh.plotting import figure, show, output_file


import numpy as np
import scipy
from scipy import ndimage
#
# from scipy import io as scio
# from scipy import misc
# from scipy.misc import toimage
# from scipy.misc import imsave, imshow
# import scipy.io as sio

import skimage.io as skio
# from skimage.util import img_as_float
# from skimage.filters import gabor_kernel

import imageio

# from PIL import Image


def img_scale_4plot_nsave(img_2scale, xsize, ysize):

    trim_fac = 0.05;

    x_arr = np.reshape(img_2scale, (1, xsize*ysize))
    x_arr_sorted = np.sort(x_arr)
    x_arr_trim_lth = np.int(xsize*ysize*trim_fac)
    x_arr_trimmed_min = np.mean(x_arr_sorted[0 : x_arr_trim_lth])
    trim_upper_idx = (xsize*ysize - x_arr_trim_lth)
    x_arr_sorted = x_arr_sorted.reshape(-1)
    x_arr_trimmed_max = np.mean(x_arr_sorted[trim_upper_idx:])

    img_2scale = np.clip(img_2scale, x_arr_trimmed_min, x_arr_trimmed_max)
    img_sf = 32000.0/(img_2scale.max() - img_2scale.min() + 1.0)

    img_offset = img_2scale.min()
    img_scaled = np.multiply((img_2scale - img_offset), img_sf)
    return img_scaled


def uk_xray(out_file_dir, img_name_os, img_ext, res_file_dir, sm_fac, order, sm_fac_l2l, sm_fac_l2l_order, prefix, roi_size_beg, roi_size_max, roi_size_incr, div_offsetx,
                write_freq, log_sw, log_off, x_off, y_off, rescaling, roi_scaling, mean_min, mad_vpk2pk_scaling, rgb_idx, dbg_img_inter, img_trim_fac):


    dir2_img = out_file_dir + img_name_os + img_ext

    fname_var_str = "\n reading:  " + dir2_img
    print(fname_var_str)

    oc_img_raw_z = skio.imread(dir2_img)
    oc_img_raw_shape = oc_img_raw_z.shape

    if (len(oc_img_raw_shape)== 3):
        oc_img_raw_z = oc_img_raw_z[:,:,rgb_idx]
        
    oc_img_raw = oc_img_raw_z.astype(float)

    oc_img_shape = oc_img_raw.shape
    oc_img_xsize = oc_img_shape[0]
    oc_img_ysize = oc_img_shape[1]

    oc_img_xsize_off = np.int(oc_img_xsize*img_trim_fac)
    oc_img_ysize_off = np.int(oc_img_ysize*img_trim_fac)
    oc_img_raw = oc_img_raw[oc_img_xsize_off : oc_img_xsize - oc_img_xsize_off, oc_img_ysize_off : oc_img_ysize - oc_img_ysize_off]

    oc_img_shape = oc_img_raw.shape
    oc_img_xsize = oc_img_shape[0]
    oc_img_ysize = oc_img_shape[1]

    # img_pyr = skimage.transform.pyramid_gaussian(oc_img_raw, max_layer=5, downscale=2, sigma=None, order=1, mode='reflect', cval=0, multichannel='false')

    write_ct = 0

    for gauss_var in range(1, 2, 1):

        gauss_var_str = "\n gauss_var:  " + str(gauss_var)
        print(gauss_var_str)

        roi_scaling_str = "\n roi_scaling:  " + str(roi_scaling)
        print(roi_scaling_str)

        meanref_str = "\n meanref:  " + str(mean_min)
        print(meanref_str)

        mad_vpk2pk_scaling_str = "\n mad_vpk2pk_scaling:  " + str(mad_vpk2pk_scaling)
        print(mad_vpk2pk_scaling_str)

        log_flg_str = "\n log sw:  " + str(log_sw)
        print(log_flg_str)

        for juno_rgb in range(0, 1, 1):

            rgb_var_str = "    rgb_var:  " + str(juno_rgb)
            print(rgb_var_str)

            # oc_img = oc_img_raw[:, :, juno_rgb]
            oc_img = oc_img_raw

            if(log_sw == 1):
                oc_img_nz = np.add(oc_img, log_off)
                oc_img = np.log(oc_img_nz)


            img_dtype = oc_img.dtype
            img_shape = oc_img.shape

            # oc_img_arr = np.reshape(oc_img, (1,np.product(oc_img.shape)))
            # oc_img_fl = oc_img_arr.flatten()

            oc_img_acc_mad_rel2_pk2pk_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_acc_pk2pk_rel2_mad_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_inv_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_acc_mad_sc_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_sc_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_acc_mad_inv_sc_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_inv_sc_isum = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_var_acc = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_mean = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_var = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_mean_roi = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
            oc_img_mad_var_roi = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')

            k = 0
            roi_size_idx = -1
            oc_img_i = oc_img

            for roi_size in range(roi_size_beg, roi_size_max, roi_size_incr):
                roi_size_str =  gauss_var_str + " roi size:  " + str(roi_size)
                print(roi_size_str)

                roi_size_idx = roi_size_idx +1
                roi_offset = int((roi_size -1)/2)

                oc_img_acc_mad_rel2_pk2pk = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
                oc_img_acc_pk2pk_rel2_mad = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
                oc_img_acc_mad_sc = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
                oc_img_acc_mad_inv_sc = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')

                oc_img_mad = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
                oc_img_mad_inv = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
                oc_img_mad_sc = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')
                oc_img_mad_inv_sc = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')

                oc_img_var = np.zeros((oc_img_xsize, oc_img_ysize), dtype='float')

                # Smoothing in proportion to roi size:
                #
                gauss_var_l = roi_size * sm_fac
                gauss_var_str = "\n gauss_var smoothing factor this ROI size:  " + str(gauss_var_l)
                print(gauss_var_str)

                if gauss_var_l > 0:
                    oc_img_curr_scale = np.copy(oc_img_i)
                    scipy.ndimage.filters.gaussian_filter(oc_img_i, gauss_var_l, order, output=oc_img)

                # oc_img_diff_pyr = np.subtract(oc_img_curr_scale, oc_img)

                for i in range(0, oc_img_xsize - 1 - roi_size + 1):

                    if (i % 20) == 0:
                        i_str = roi_size_str + " i:  " + str(i)
                        print(i_str)

                    for j in range(0, oc_img_ysize - 1 - roi_size + 1):

                        roi = oc_img[i:(i + roi_size), j:(j + roi_size)]

                        # s_var = np.sqrt(np.var(roi))
                        # s_std = np.std(roi)
                        smean = np.mean(roi)

                        # s_var = np.var(roi)
                        # s_mad = np.var(roi)    # Use variance now instead of mad
                        s_mad = np.mean(np.fabs(roi - smean))

                        # oc_img_var[i, j] = s_mad

                        smin = roi.min()
                        smax = roi.max()

                        # s_mad_risc = s_mad / roi_size           # ROI inverse scaling

                        s_pk2pk = smax - smin
                        div_offset = s_pk2pk / 1000.0

                        # sfac_std = s_std / roi_size   #  ((s_pk2pk + 0.1))    # *roi_size)

                        if(mean_min == 0) :
                            roi_pk2pk_rescaled = np.subtract(roi, smin)    # smean
                        else :
                            roi_pk2pk_rescaled = np.subtract(roi, smean)    # smean

                        if(rescaling == 1):
                            offset_l = smax / 100000.0
                            roi_pk2pk_rescaled = np.multiply(roi_pk2pk_rescaled, 1000.0/(smax - smin + offset_l + 1))

                        if(roi_scaling == 1):
                            roi_pk2pk_rescaled = np.multiply(roi_pk2pk_rescaled, (1.0 / roi_size) )

                        # Add peak-to-peak / var factor:
                        #   - original scale peak-to-peak:
                        roi_orig = oc_img_curr_scale[i:(i + roi_size), j:(j + roi_size)]
                        smin_orig = roi_orig.min()
                        smax_orig = roi_orig.max()
                        s_pk2pk_orig = smax_orig - smin_orig
                        # s_mad = s_mad / (s_pk2pk_orig + 1.0)

                        if(mad_vpk2pk_scaling == 1):
                            s_mad = s_mad / (s_pk2pk + 1.0)

                        # s_mad_rescaled = np.std(roi_pk2pk_rescaled)    # Use variance now instead of mad

                        # oc_img_var[i + roi_offset, j + roi_offset] = s_mad;

                        roi_offset = int(roi_offset)
                        io = int(i + roi_offset)
                        jo = int(j + roi_offset)
                        oc_img_var[io, jo] = s_mad;

                        # oc_img_var_acc[i:(i + roi_size + 1), j:(j + roi_size + 1)] = np.add(oc_img_var_acc[i:(i + roi_size + 1), j:(j + roi_size + 1)], roi_pk2pk_rescaled)

                        # scale_fac = 1.0   # s_mad/(s_pk2pk + div_offset)
                        # roi_mad_rel2_pk2pk = np.multiply(roi_pk2pk_rescaled,  scale_fac)   # Local mad w/ respect to local pk-to-pk scaling  -- Junocam upload type #2
                        # oc_img_acc_mad_rel2_pk2pk[i:(i + roi_size), j:(j + roi_size)] = np.add(oc_img_acc_mad_rel2_pk2pk[i:(i + roi_size), j:(j + roi_size)], roi_mad_rel2_pk2pk)
                        # oc_img_mad[i:(i + roi_size ), j:(j + roi_size)] = np.add(oc_img_mad[i:(i + roi_size), j:(j + roi_size)], scale_fac)  # roi_pk2pk_mad_rel2_pk2pk
                        #
                        # scale_fac = s_pk2pk/(s_mad + div_offset + div_offsetx)
                        # roi_pk2pk_rel2_mad = np.multiply(roi_pk2pk_rescaled, scale_fac)   # Local mad w/ respect to local pk-to-pk scaling  -- Junocam upload type #2
                        # oc_img_acc_pk2pk_rel2_mad[i:(i + roi_size), j:(j + roi_size)] = np.add(oc_img_acc_pk2pk_rel2_mad[i:(i + roi_size), j:(j + roi_size)], roi_pk2pk_rel2_mad)
                        # oc_img_mad_inv[i:(i + roi_size), j:(j + roi_size)] = np.add(oc_img_mad_inv[i:(i + roi_size), j:(j + roi_size)], scale_fac )  # roi_pk2pk_mad_rel2_pk2pk



                        # scale_fac = s_mad
                        roi_mad_sc = np.multiply(roi_pk2pk_rescaled, s_mad)  # Local mad w/ respect to local pk-to-pk scaling  -- Junocam upload type #2
                        oc_img_acc_mad_sc[i:(i + roi_size), j:(j + roi_size)] = np.add(oc_img_acc_mad_sc[i:(i + roi_size), j:(j + roi_size)], roi_mad_sc)
                        oc_img_mad_sc[i:(i + roi_size), j:(j + roi_size)] = np.add(oc_img_mad_sc[i:(i + roi_size), j:(j + roi_size)], s_mad )

                        oc_img_acc_mad_inv_sc[i + roi_offset, j + roi_offset] = np.add(oc_img_acc_mad_inv_sc[i + roi_offset, j + roi_offset], roi_mad_sc[roi_offset, roi_offset])
                        oc_img_mad_inv_sc[i + roi_offset, j + roi_offset] = np.add(oc_img_mad_inv_sc[i + roi_offset, j + roi_offset], s_mad )

                        if(roi_size_idx == 0):
                            oc_img_mad_mean[i + roi_offset, j + roi_offset] = s_mad    # Init running mean & variance -- variance is already initialized (== 0)
                        else:
                            s_mad_t = oc_img_mad_mean[i + roi_offset, j + roi_offset]
                            s_mad_upd = s_mad_t + (s_mad - s_mad_t) / (roi_size_idx +1)
                            oc_img_mad_mean[i + roi_offset, j + roi_offset] = s_mad_upd

                            s_var_t = oc_img_mad_var[i + roi_offset, j + roi_offset]
                            s_var_t = s_var_t + (s_mad - s_mad_t)*(s_mad - s_mad_upd)
                            oc_img_mad_var[i + roi_offset, j + roi_offset] = s_var_t


                        # Running mean & var on entire ROI:
                        #
                        if(roi_size_idx == 0):
                            oc_img_mad_mean_roi[i:(i + roi_size), j:(j + roi_size)] = roi    # Init running mean & variance -- variance is already initialized (== 0)
                        else:
                            s_mad_t = oc_img_mad_mean_roi[i:(i + roi_size), j:(j + roi_size)]
                            s_mad_upd = s_mad_t + (roi - s_mad_t) / (roi_size_idx +1)
                            oc_img_mad_mean_roi[i:(i + roi_size), j:(j + roi_size)] = s_mad_upd

                            s_var_t = oc_img_mad_var_roi[i:(i + roi_size), j:(j + roi_size)]
                            # s_var_t = s_var_t + (s_mad - s_mad_t)*(s_mad - s_mad_upd)
                            s_var_t = np.multiply((s_mad - s_mad_t), (s_mad - s_mad_upd))
                            oc_img_mad_var_roi[i:(i + roi_size), j:(j + roi_size)] = s_var_t

                # scale_fac = 1.0 / (s_mad + div_offset)
                        # roi_mad_inv_sc = np.multiply(roi_pk2pk_rescaled, scale_fac)  # Local mad w/ respect to local pk-to-pk scaling  -- Junocam upload type #2
                        # oc_img_acc_mad_inv_sc[i:(i + roi_size + 1), j:(j + roi_size + 1)] = np.add(oc_img_acc_mad_inv_sc[i:(i + roi_size + 1), j:(j + roi_size + 1)], roi_mad_inv_sc)
                        # oc_img_mad_inv_sc[i:(i + roi_size + 1), j:(j + roi_size + 1)] = np.add(oc_img_mad_inv_sc[i:(i + roi_size + 1), j:(j + roi_size + 1)], scale_fac)

                roi_size_str =  gauss_var_str + " finit roi size:  " + str(roi_size)
                print(roi_size_str)

                oc_img_var_acc = np.add(oc_img_var_acc, oc_img_var)


                # Smooth each ROI image and associated ROI scaling image prior to accumulation over scale:
                #
                gauss_var_l2l = sm_fac_l2l * roi_size

                if(gauss_var_l2l > 0):
                    scipy.ndimage.filters.gaussian_filter(oc_img_acc_mad_rel2_pk2pk, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_acc_mad_rel2_pk2pk)
                oc_img_acc_mad_rel2_pk2pk_isum = np.add(oc_img_acc_mad_rel2_pk2pk_isum, oc_img_acc_mad_rel2_pk2pk)

                if(dbg_img_inter > 0):
                    plt.imshow(oc_img_acc_mad_rel2_pk2pk_isum, 'gray')
                    # bleffle = input("oc_img_acc_mad_rel2_pk2pk_isum")
                    #imshow(oc_img_acc_mad_rel2_pk2pk_isum)

                    # img = oc_img_acc_mad_rel2_pk2pk_isum
                    # plot = figure(plot_width=img.shape[0], plot_height=img.shape[1], x_range=[0, img.shape[0]], y_range=[0, img.shape[1]])
                    # img_ht = img.shape[0]
                    # img_width = img.shape[1]
                    # plot.image(image=[img], x=[0], y=[0], dw=[img_ht], dh=[img_width])
                    # show(plot)

                if(gauss_var_l2l > 0):
                    scipy.ndimage.filters.gaussian_filter(oc_img_mad, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_mad)
                oc_img_mad_isum = np.add(oc_img_mad_isum, oc_img_mad)
                if(dbg_img_inter > 0):
                    plt.imshow(oc_img_acc_mad_rel2_pk2pk_isum, 'gray')
                    # img = oc_img_mad_isum
                    # plot = figure(plot_width=img.shape[0], plot_height=img.shape[1], x_range=[0, img.shape[0]], y_range=[0, img.shape[1]])
                    # plot.image(image=[numpy.flipud(img)], x=[0], y=[0], dw=[img.shape[0]], dh=[img.shape[1]])

                if(gauss_var_l2l > 0):
                    scipy.ndimage.filters.gaussian_filter(oc_img_acc_pk2pk_rel2_mad, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_acc_pk2pk_rel2_mad)
                oc_img_acc_pk2pk_rel2_mad_isum = np.add(oc_img_acc_pk2pk_rel2_mad_isum, oc_img_acc_pk2pk_rel2_mad)
                if(dbg_img_inter > 0):
                    plt.imshow(oc_img_acc_pk2pk_rel2_mad_isum, 'gray')

                if(gauss_var_l2l > 0):
                    scipy.ndimage.filters.gaussian_filter(oc_img_mad_inv, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_mad_inv)
                oc_img_mad_inv_isum = np.add(oc_img_mad_inv_isum, oc_img_mad_inv)
                if(dbg_img_inter > 0):
                    # bleffle = input("oc_img_mad_inv_isum")
                    plt.imshow(oc_img_mad_inv_isum, 'gray')

                if (gauss_var_l2l > 0):
                    scipy.ndimage.filters.gaussian_filter(oc_img_acc_mad_sc, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_acc_mad_sc)
                oc_img_acc_mad_sc_isum = np.add(oc_img_acc_mad_sc_isum, oc_img_acc_mad_sc)
                if(dbg_img_inter > 0):
                    # bleffle = input("oc_img_acc_mad_sc_isum")
                    plt.imshow(oc_img_acc_mad_sc_isum, 'gray')

                if (gauss_var_l2l > 0):
                    scipy.ndimage.filters.gaussian_filter(oc_img_mad_sc, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_mad_sc)
                oc_img_mad_sc_isum = np.add(oc_img_mad_sc_isum, oc_img_mad_sc)
                if(dbg_img_inter > 0):
                    # bleffle = input("oc_img_mad_sc_isum")
                    plt.imshow(oc_img_mad_sc_isum, 'gray')

                if (gauss_var_l2l > 0):
                    scipy.ndimage.filters.gaussian_filter(oc_img_acc_mad_inv_sc, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_acc_mad_inv_sc)
                oc_img_acc_mad_inv_sc_isum = np.add(oc_img_acc_mad_inv_sc_isum, oc_img_acc_mad_inv_sc)
                if (dbg_img_inter > 0):
                    # bleffle = input("oc_img_acc_mad_inv_sc_isum")
                    plt.imshow(oc_img_acc_mad_inv_sc_isum, 'gray')

                if (gauss_var_l2l > 0):
                        scipy.ndimage.filters.gaussian_filter(oc_img_mad_inv_sc, gauss_var_l2l, sm_fac_l2l_order, output=oc_img_mad_inv_sc)
                        oc_img_mad_inv_sc_isum = np.add(oc_img_mad_inv_sc_isum, oc_img_mad_inv_sc)
                        plt.imshow(oc_img_mad_inv_sc_isum, 'gray')

                write_ct = write_ct +1
                if (write_ct == write_freq):

                    write_ct = 0

                    # oc_img_acc_mad_rel2_pk2pk = oc_img_mad_var

                    # oc_img_t = np.divide(oc_img_mad_var, (oc_img_mad_mean + 1))
                    oc_img_t = np.divide(oc_img_mad_var_roi, (oc_img_mad_mean_roi + 1))
                    oc_img_acc_mad_rel2_pk2pk = oc_img_t
                    oc_img_t = np.add(oc_img_t, oc_img_t.mean()/1000.0)
                    oc_img_t = np.log(oc_img_t +1)

                    oc_img_acc_pk2pk_rel2_mad = np.multiply(oc_img_acc_mad_inv_sc_isum, oc_img_t)

                    # oc_img_acc_pk2pk_rel2_mad = np.add(oc_img_acc_pk2pk_rel2_mad, 1.0)
                    # oc_img_acc_pk2pk_rel2_mad = np.log(oc_img_acc_pk2pk_rel2_mad)

                    # # Scale for output and generate output images at this scale:
                    # #
                    # # oc_img_madx = np.clip(oc_img_mad_isum, oc_img_mad_isum.max()/1000.0, oc_img_mad_isum.max())
                    # # oc_img_acc_pk2pk_std_sz = np.divide(oc_img_acc_mad_rel2_pk2pk_isum, oc_img_madx)
                    # oc_img_enh_pk2pk_std_sz = img_scale_4plot_nsave(oc_img_acc_mad_rel2_pk2pk, oc_img_xsize, oc_img_ysize)
                    # img_fname_strx = out_file_dir + prefix + "mad_mean_" + img_name_os +  "_roi" + str(roi_size) + "_gvar" + str(gauss_var) + "_rgb" + str(juno_rgb) + ".png"
                    # # imsave(img_fname_strx, oc_img_enh_pk2pk_std_sz)
                    # # imageio.imwrite(img_fname_strx, oc_img_enh_pk2pk_std_sz)
                    # skio.imsave(img_fname_strx, oc_img_enh_pk2pk_std_sz)

                    # oc_img_madx_inv = np.clip(oc_img_mad_inv_isum, oc_img_mad_inv_isum.max()/1000.0, oc_img_mad_inv_isum.max())
                    # oc_img_acc_pk2pk_std_sz = np.divide(oc_img_acc_pk2pk_rel2_mad_isum, oc_img_madx_inv)
                    # oc_img_enh_pk2pk_std_sz = img_scale_4plot_nsave(oc_img_acc_pk2pk_rel2_mad, oc_img_xsize, oc_img_ysize)


                    # # # Raw scaled ROI sum:
                    # # Gaussian pyramid difference images:
                    # # oc_img_enh_pk2pk_std_sz = img_scale_4plot_nsave(oc_img_mad_sc_isum, oc_img_xsize, oc_img_ysize)
                    # oc_img_enh_pk2pk_std_sz = img_scale_4plot_nsave(oc_img_diff_pyr, oc_img_xsize, oc_img_ysize)
                    # img_fname_strx = out_file_dir + prefix + "gauss_pyr_" + img_name_os +  "_roi" + str(roi_size) + "_gvar" + str(gauss_var) + "_rgb" + str(juno_rgb) + ".jpg"
                    # imageio.imwrite(img_fname_strx, oc_img_enh_pk2pk_std_sz)
                    # imageio.imwrite(img_fname_strx, oc_img_enh_pk2pk_std_sz)


                    oc_img_mad_scx = np.clip(oc_img_mad_sc_isum, oc_img_mad_sc_isum.max()/1000.0, oc_img_mad_sc_isum.max())
                    oc_img_acc_pk2pk_std_sz = np.divide(oc_img_acc_mad_sc_isum, oc_img_mad_scx)
                    oc_img_enh_pk2pk_std_sz = img_scale_4plot_nsave(oc_img_acc_pk2pk_std_sz, oc_img_xsize, oc_img_ysize)
                    img_fname_strx = res_file_dir + prefix + "mad_sc_roi_" + img_name_os +  "_roi" + str(roi_size) + "_gvar" + str(gauss_var)  + "_rgb" + str(juno_rgb) + ".png"
                    fname_var_stro = "\n writing:  " + img_fname_strx
                    print(fname_var_stro)
                    skio.imsave(img_fname_strx, oc_img_enh_pk2pk_std_sz)

                    oc_img_mad_inv_scx = np.clip(oc_img_mad_inv_sc_isum, oc_img_mad_inv_sc_isum.max()/1000.0, oc_img_mad_inv_sc_isum.max())
                    oc_img_acc_pk2pk_std_sz = np.divide(oc_img_acc_mad_inv_sc_isum, oc_img_mad_inv_scx)
                    oc_img_enh_pk2pk_std_sz = img_scale_4plot_nsave(oc_img_acc_pk2pk_std_sz, oc_img_xsize, oc_img_ysize)
                    img_fname_strx = res_file_dir + prefix + "mad_sc_ctr_" + img_name_os +  "_roi" + str(roi_size) + "_gvar" + str(gauss_var)  + "_rgb" + str(juno_rgb) + ".png"
                    fname_var_stro = "\n writing:  " + img_fname_strx
                    print(fname_var_stro)
                    skio.imsave(img_fname_strx, oc_img_enh_pk2pk_std_sz)

        bleffy = 14

    jemble = 12

