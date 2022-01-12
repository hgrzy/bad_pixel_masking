import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
class bad_pix:
    def __init__(self, fits_path):
        f = fits.open(fits_path)
        self.stack_pixels = f[0].data
        self.header = f[0].header
        self.bad_pixel_mask = np.empty_like(f[0].data[0])
        #self.ser_num = stuff from header
        f.close()
    

    @property
    def pixel_x(self):
        return np.size(self.stack_pixels, 1)
    
    @property
    def pixel_y(self):
        return np.size(self.stack_pixels, 2)

    @property
    def num_frames(self):
        return np.size(self.stack_pixels, 0)

    def Gaussian(self,x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))



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

    def pixel_mask(self, x0, sigma, sens_crit):
        self.bad_pixel_mask = (self.pixel_rms <= x0 + sens_crit * sigma) & (self.pixel_rms >= x0 - sens_crit * sigma) # array with true for within criterion and false if not

   
    #save_path = os.path.abspath(r'C:\Users\mow5307\Documents\Hannah_masking_temporary\.vscode\hist_outputs')
    def histograms(self, save_path, run_num, sens_crit, bp_Obj):
        #bp_Obj is a bad pixel object, e.g. bad_pix_dict['Run01']

        if hasattr(self, "pixel_rms"):

            #Fit Gaussian Curve
            y, x, _ = plt.hist(self.pixel_rms.flatten(), bins=500, range = [-5,30])

            xdata = np.asarray(x[:-1])
            ydata = np.asarray(y)

            # Plot out the current state of the data and model
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(xdata, ydata)

            # Executing curve_fit on data
            popt, pcov = curve_fit(bp_Obj.Gaussian, xdata, ydata) #popt is best fit values for parameters of given model (Gaussian)

            #make plot
            yfit = self.Gaussian(xdata, popt[0], popt[1], popt[2])
            ax.plot(xdata, yfit, c='r', label='Best fit')
            ax.legend()
            Gaussfit_save = save_path + "\\" + run_num + "\\Gauss_fit" 
            os.makedirs(Gaussfit_save)
            plt.savefig(Gaussfit_save)
            print("Gaussian fit with saved to %s"%Gaussfit_save)
            fig.savefig('Gaussian_fit.png')

        
            #post-masking histogram
            #get bad_pixel mask
            plt.figure('Post-Masked Histogram')

            
            #popt values are the pixels that are within the gaussian, creating a bad pixel
            #mask for pixels that are within the gaussian but outside of the given sigma
            self.pixel_mask(popt[1], popt[2], sens_crit)

            good_pix = self.pixel_rms * self.bad_pixel_mask.astype(int)
            plt.hist(good_pix.flatten(), bins=500, range = [-5,30])
            axes = plt.gca()
            axes.set_xlim([-5,30])
            axes.set_ylim([0,13000])

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
        ax = plt.gca()
        ax.Colorscale = 'log'
        plt.imshow(arr)
        plt.savefig(fpath)

        #post-mask
        plt.figure('Post-masked frame')
        fpath_post = save_path + "\\" + run_num + "\\frame" + str(frame_num) + '\\post_mask'
        os.makedirs(fpath_post)
        os.chdir(fpath_post)
        arr2 = self.stack_pixels[frame_num] * self.bad_pixel_mask
        #plt.imsave(fpath_post, arr2)
        ax = plt.gca()
        ax.Colorscale = 'log'
        plt.imshow(arr2)
        plt.savefig(fpath_post)

        #pixel_mask
        fpath_bp = save_path + "\\" + run_num + "\\frame" + str(frame_num) + '\\bad_pix'
        os.makedirs(fpath_bp)
        os.chdir(fpath_bp)
        arr3 = self.bad_pixel_mask
        #plt.imsave(fpath_bp, arr3)
        ax = plt.gca()
        ax.Colorscale = 'log'
        plt.imshow(arr3)
        plt.savefig(fpath_bp)



        

