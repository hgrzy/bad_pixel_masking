import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
class bad_pix:
    def __init__(self, fits_path):
        f = fits.open(fits_path)
        self.stack_pixels = f[0].data
        self.header = f[0].header
        self.bad_pixel_mask = np.empty_like(f[0].data[0])
        #self.ser_num = stuff from header
        f.close()
    

    # @property
    # def pixel_x(self):
    #     return np.size(self.stack_pixels, 1)
    
    @property
    def pixel_y(self):
        return np.size(self.stack_pixels, 2)

    @property
    def num_frames(self):
        return np.size(self.stack_pixels, 0)

    


    def pixel_values(self):
        self.pixel_rms    = np.sqrt((1/self.num_frames)*np.sum(np.square(self.stack_pixels), axis = 0)) #rms for each pixel across frames
        self.pixel_mean   = np.mean(self.stack_pixels, axis =0) #mean of pixel across frames
        self.pixel_median = np.median(self.stack_pixels, axis =0) #median of pixel across frames
        self.pixel_std    = np.std(self.stack_pixels, axis = 0) #standard deviation of pixel across frames
    
    def detector_values(self):
        self.det_rms = np.sqrt((1/(self.num_frames*self.pixel_x*self.pixel_y))*np.sum(np.square(self.stack_pixels)))
        self.det_mean = np.mean(self.stack_pixels)
        self.det_median = np.median(self.stack_pixels)
        self.det_std = np.std(self.stack_pixels)

    def pixel_mask(self, sens_crit):
        self.bad_pixel_mask = (self.pixel_rms <= np.mean(self.pixel_rms.flatten()) + sens_crit * np.std(self.pixel_rms.flatten())) & (self.pixel_rms >= np.mean(self.pixel_rms.flatten()) - sens_crit * np.std(self.pixel_rms.flatten())) # array with true for within criterion and false if not

   
    #save_path = os.path.abspath(r'C:\Users\mow5307\Documents\Hannah_masking_temporary\.vscode\hist_outputs')
    def histograms(self, save_path, run_num, sens_crit):

        if hasattr(self, "pixel_rms"):

            #pre-masking histogram
            plt.figure('Pre-Masked Histogram')
            plt.hist(self.pixel_rms.flatten(),bins=500,range = [-5, 30]);
            plt.vlines(np.mean(self.pixel_rms.flatten()), 0, 10000, color = 'r')
            plt.vlines(np.median(self.pixel_rms.flatten()), 0, 15000)
            plt.vlines(np.mean(self.pixel_rms.flatten()) + sens_crit * np.std(self.pixel_rms.flatten()), 0, 15000, color = 'green')
            plt.vlines(np.mean(self.pixel_rms.flatten()) - sens_crit * np.std(self.pixel_rms.flatten()), 0, 15000, color = 'green')

            pre_mask_save = save_path + "\\" + run_num + "\\pre_mask" 
            os.makedirs(pre_mask_save)
            plt.savefig(pre_mask_save)
            print("Pre-mask with rms lines histogram saved to %s"%pre_mask_save)


        
            #post-masking histogram
            #get bad_pixel mask
            plt.figure('Post-Masked Histogram')

            self.pixel_mask(sens_crit)

            good_pix = self.pixel_rms * self.bad_pixel_mask.astype(int)
            plt.hist(good_pix.flatten(), bins=500, range = [-5,30])

            post_mask_save = save_path + "\\" + run_num + "\\post_mask"
            os.makedirs(post_mask_save)
            plt.savefig(post_mask_save)
            print("Post-mask histogram saved to %s"%post_mask_save)
        else:
            print("Pixel values have not been evaluated yet. Please run pixel_values() first.")


    def plots(self, save_path, run_num, frame_num):
        #pre-mask
        plt.figure('Pre-masked frame')
        fpath = save_path + "\\" + run_num + "\\frame" + str(frame_num) + '\\aFile'
        os.makedirs(fpath)
        os.chdir(fpath)
        arr = self.stack_pixels[frame_num]
        #plt.imsave(fpath, arr, format = 'png')
        plt.imshow(arr)
        plt.savefig(fpath)

        #post-mask
        fpath_post = save_path + "\\" + run_num + "\\frame" + str(frame_num) + '\\post_mask.png'
        arr2 = self.stack_pixels[frame_num] * self.bad_pixel_mask
        #plt.imsave(fpath_post, arr2)

        #pixel_mask
        fpath_bp = save_path + "\\" + run_num + "\\frame" + str(frame_num) + '\\bad_pix.png'
        arr3 = self.bad_pixel_mask
        #plt.imsave(fpath_bp, arr3)



        

