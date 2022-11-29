import numpy as np
from scipy.optimize import lsq_linear
from skimage.transform import downscale_local_mean


class UBDF:
    """
    Implementation of the Unmixing-Based Data Fusion (UBDF) method [1].

    UBDF requires a fine-resolution image (hereafter "fine image") at the known date (t1), and a coarse-resolution image
    (hereafter "coarse image") at the prediction date (t2) as the input. UBDF outputs the predicted fine image at t2.

    There are two basic assumptions of the UBDF method: (1) the reflectance of a coarse pixel is the linear mixture of
    the reflectances of each endmember; (2) no land cover change occurs between t1 and t2.

    UBDF consists of four main procedures:
    (1) Classify the fine image at t1 to get the land cover classification map (i.e., define the endmembers).
    (2) Calculate the fractions of the land cover classes inside each coarse pixel.
    (3) Unmix the coarse pixels within a moving window to get the reflectance of each land cover class.
    (4) Predict the fine image at t2 using the unmixed reflectances and the classification map.

    Reference:
    [1] Zurita-Milla, R., Clevers, J. G. P. W., & Schaepman, M. E. (2008). Unmixing-based landsat TM and MERIS FR data
    fusion. IEEE Geoscience and Remote Sensing Letters, 5, 453–457. http://dx.doi.org/10.1109/LGRS.2008.919685.

    NOTE:
    This implementation was finished by Houcai Guo (personal webpage: houcaiguo.github.io). If you have any problems,
    please feel free to contact me.
    """

    def __init__(self, C_t2, F_t1_class, class_num, scale_factor=16, win_size=3):
        """
        Initialization of UBDF.

        Parameters
        ----------
        C_t2 : array_like
            Coarse image at t2, (C_row, C_col, C_channel) shaped.
        F_t1_class : array_like
            Classification map of the fine image at t1.
        class_num : int
            Number of classes.
        scale_factor : int
            Scale factor of the coarse-fine image pair, for Sentinel-3 and Sentinel-2, the factor is 300//10=30.
        win_size : int
            Size of the moving window in the unmixing process.
        """
        self.C_t2 = C_t2
        # rescale C_t2 if it has the same shape as F_t1
        # self.C_t2 = downscale_local_mean(C_t2, factors=(scale_factor, scale_factor, 1))
        self.F_t1_class = F_t1_class
        self.class_num = class_num
        self.scale_factor = scale_factor
        self.win_size = win_size

    def calculate_fractions(self):
        """
        Calculate the fractions of the land cover classes inside each coarse pixel.

        Returns
        -------
        C_fractions : array_like, (C_row, C_col, class_num) shaped
            Fractions of the land cover classes.
        """
        C_row = self.C_t2.shape[0]
        C_col = self.C_t2.shape[1]
        C_fractions = np.zeros(shape=(C_row, C_col, self.class_num), dtype=np.float32)
        for C_row_idx in range(C_row):
            for C_col_idx in range(C_col):
                F_class_pixels = self.F_t1_class[C_row_idx * self.scale_factor:(C_row_idx + 1) * self.scale_factor,
                                 C_col_idx * self.scale_factor:(C_col_idx + 1) * self.scale_factor, :]
                for class_idx in range(self.class_num):
                    pixel_num = np.count_nonzero(F_class_pixels == class_idx)
                    C_fractions[C_row_idx, C_col_idx, class_idx] = pixel_num / (self.scale_factor * self.scale_factor)

        return C_fractions

    def unmix_window(self, C_pixels, C_fractions, lower_bound, upper_bound):
        """
        Use a constrained least squares method to unmix the coarse pixel to get the reflectance of the each class
        in the predicted image covered by the central coarse pixel of the moving window. The solved reflectances should
        be positive and not exceed the reflectance of the reflectance of coarse pixel.

        Parameters
        ----------
        C_pixels : array_like, (win_size**2, 1) shaped
            Reflectances of the coarse pixels in the moving window.
        C_fractions : array_like, (win_size**2, class_sum) shaped
            Fractions of all the classes inside each coarse pixel.
        lower_bound : np.float32
            Lower bound of the least squares method.
        upper_bound : np.float32
            Upper bound of the least squares method.
        Returns
        -------
        result : array_like, (class_num, 1) shaped
            The unmixed reflectances.
        """
        result = lsq_linear(C_fractions, C_pixels,
                            bounds=(lower_bound, upper_bound), method="bvls").x

        return result

    def unmixing_based_data_fusion(self):
        """
        Unmxing-Based Data Fusion.

        Returns
        -------
        F_t2_predict : array_like
            The predicted fine image at t2.
        """
        F_t2_predict = np.empty(shape=(self.F_t1_class.shape[0], self.F_t1_class.shape[1], self.C_t2.shape[2]),
                        dtype=self.C_t2.dtype)

        # calculate the fractions
        C_fractions = self.calculate_fractions()

        # pad the coarse image at t2 and the fractions map since the unmixing process is based on a local window
        C_t2_pad = np.pad(self.C_t2, pad_width=((self.win_size // 2, self.win_size // 2),
                                                (self.win_size // 2, self.win_size // 2), (0, 0)),
                          mode="reflect")
        C_fractions_pad = np.pad(C_fractions, pad_width=((self.win_size // 2, self.win_size // 2),
                                                         (self.win_size // 2, self.win_size // 2), (0, 0)),
                                 mode="reflect")

        # predict the fine image band-by-band
        for band_idx in range(self.C_t2.shape[2]):
            band = C_t2_pad[:, :, band_idx]
            upper_bound = np.max(band)
            for row_idx in range(self.C_t2.shape[0]):
                for col_idx in range(self.C_t2.shape[1]):
                    # coarse pixels and fractions inside the moving window
                    C_pixels_win = band[row_idx:row_idx + self.win_size, col_idx:col_idx + self.win_size]
                    C_fractions_win = C_fractions_pad[row_idx:row_idx + self.win_size,
                                      col_idx:col_idx + self.win_size, :]

                    # unmixing using the local window
                    F_class_values = self.unmix_window(C_pixels_win.reshape((self.win_size ** 2,)),
                                                       C_fractions_win.reshape(self.win_size ** 2, self.class_num),
                                                       lower_bound=0, upper_bound=upper_bound)

                    # class information of the central coarse pixel inside the window
                    C_class = self.F_t1_class[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                              col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor]

                    # assign the reflectances of the fine pixels inside current coarse pixel class-by-class
                    for class_idx in range(self.class_num):
                        class_mask = np.where(C_class == class_idx, True, False).squeeze()
                        F_t2_predict[row_idx * self.scale_factor:(row_idx + 1) * self.scale_factor,
                        col_idx * self.scale_factor:(col_idx + 1) * self.scale_factor, band_idx][class_mask] = \
                            F_class_values[class_idx]

        return F_t2_predict

