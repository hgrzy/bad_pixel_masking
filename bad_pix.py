import numpy as np
from astropy.io import fits
from astropy.io.fits import Header
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
        run_ind = fits_path.index('Run') #find index in path where this occurs
        self.run_num = fits_path[run_ind : run_ind + 5]
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
    def histograms(self, save_path, sens_crit, bp_Obj):
        #bp_Obj is a bad pixel object, e.g. bad_pix_dict['Run01']

        if hasattr(self, "pixel_rms"):

            fpath = save_path + "\\" + self.run_num  + '\\histograms'
            os.makedirs(fpath)
            os.chdir(fpath)

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
            Gaussfit_save = save_path + "\\" + self.run_num + "\\histograms\\Gauss_fit" 
            plt.savefig(Gaussfit_save)
            print("Gaussian fit saved to %s"%Gaussfit_save)
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

            post_mask_save = save_path + "\\" + self.run_num + "\\histograms\\post_mask"
            plt.savefig(post_mask_save)
            print("Post-mask histogram saved to %s"%post_mask_save)
        else:
            print("Pixel values have not been evaluated yet. Please run pixel_values() first.")


    def plots(self, save_path, frame_num):
      
        plotspath = save_path + "\\" + self.run_num + "\\frame" + str(frame_num) + "\\plots"
        os.makedirs(plotspath)
        os.chdir(plotspath)

        

        plt.figure('Pre-masked frame')
        fpath = plotspath + '\\pre_mask'
        os.makedirs(fpath)
        os.chdir(fpath)
        arr = self.stack_pixels[frame_num]
        #plt.imsave(fpath, arr, format = 'png')
        ax = plt.gca()
        ax.Colorscale = 'log'
        plt.imshow(arr)
        plt.savefig(fpath)
        print("Pre-mask plot saved to %s"%fpath)

        #post-mask
        plt.figure('Post-masked frame')
        fpath_post = plotspath + '\\post_mask'
        os.makedirs(fpath_post)
        os.chdir(fpath_post)
        arr2 = self.stack_pixels[frame_num] * self.bad_pixel_mask
        #plt.imsave(fpath_post, arr2)
        ax = plt.gca()
        ax.Colorscale = 'log'
        plt.imshow(arr2)
        plt.savefig(fpath_post)
        print("Post-mask plot saved to %s"%fpath_post)

        #pixel_mask
        fpath_bp = plotspath + '\\bad_pix'
        os.makedirs(fpath_bp)
        os.chdir(fpath_bp)
        arr3 = self.bad_pixel_mask
        #plt.imsave(fpath_bp, arr3)
        ax = plt.gca()
        ax.Colorscale = 'log'
        plt.imshow(arr3)
        plt.savefig(fpath_bp)
        print("Pixel mask plot saved to %s"%fpath_bp)

    def fits(self, save_path, frame_num):

        fitspath = save_path + "\\" + self.run_num + "\\frame" + str(frame_num) + "\\fits"
        os.makedirs(fitspath)
        os.chdir(fitspath)

        #PRE-MASK

        #file name and path
        prefit_name = self.run_num + "_frame" + str(frame_num) + '_pre_mask.fits'
        fpath = fitspath + "\\" +  prefit_name

        arr = self.stack_pixels[frame_num]

        #Create empty Header object
        hdr = fits.Header()

        #Dictionary with values to put in header
        hdr_info = {'DetSer' : ('INSERT', 'Detector Serial Number'),
                    'DataObt' : ('INSERT', 'Date Data Obtained'),
                    'PixMObt' : ('INSERT', 'Date Pixel Mask Obtained'),
                    'SVR' : ('INSERT', 'Source/Voltage/Run Folder'),
                    'Frame' : ('INSERT', 'Frame Number'),
                    'Method': ('INSERT', 'INSERT'),
                    'PerctBad': ('INSERT', 'Percentage of Bad Pixels'),
                    'Mask':('False', 'Bad pixel mask applied'), 
                    'Ran_by': ('INSERT', 'Name of user'), 
                    'Vr' : (1, 'Masking Program Version')}
        
        #Update header object with info
        hdr.update(hdr_info)

        #Create fits file with header and data
        
        fits.writeto(fpath, arr, hdr)
        print("Pre-mask fits file saved to %s"%fpath)

        #POST-MASK

        #File name and path
        postfit_name = self.run_num + "_frame" + str(frame_num) + '_post_mask.fits'
        fpath_post = fitspath + "\\" + postfit_name
        
        arr_post = self.stack_pixels[frame_num] * self.bad_pixel_mask

        #Create empty Header object
        hdr_post = fits.Header()

        #Dictionary with values to put in header
        hdr_info_post = {'DetSer' : ('INSERT', 'Detector Serial Number'),
                    'DataObt' : ('INSERT', 'Date Data Obtained'),
                    'PixMObt' : ('INSERT', 'Date Pixel Mask Obtained'),
                    'SVR' : ('INSERT', 'Source/Voltage/Run Folder'),
                    'Method': ('INSERT', 'INSERT'),
                    'PerctBad': ('INSERT', 'Percentage of Bad Pixels'),
                    'Mask':('True', 'Bad pixel mask applied'), 
                    'Ran_by': ('INSERT', 'Name of user'), 
                    'Vr' : (1, 'Masking Program Version')}
        
        #Update header object with info
        hdr_post.update(hdr_info_post)

        #Create fits file with header and data
        
        fits.writeto(fpath_post, arr_post, hdr_post)
        print("Post-mask fits file saved to %s"%fpath_post)

        #MASK

        #File name and path
        maskfit_name = self.run_num + "_frame" + str(frame_num) + '_mask.fits'
        fpath_mask = fitspath + "\\" + maskfit_name
        
        arr_mask = self.bad_pixel_mask.astype(int)

        #Create empty Header object
        hdr_mask = fits.Header()

        #Dictionary with values to put in header
        hdr_info_mask = {'DetSer' : ('INSERT', 'Detector Serial Number'),
                    'DataObt' : ('INSERT', 'Date Data Obtained'),
                    'PixMObt' : ('INSERT', 'Date Pixel Mask Obtained'),
                    'SVR' : ('INSERT', 'Source/Voltage/Run Folder'),
                    'Method': ('INSERT', 'INSERT'),
                    'PerctBad': ('INSERT', 'Percentage of Bad Pixels'), 
                    'Ran_by': ('INSERT', 'Name of user'), 
                    'Vr' : (1, 'Masking Program Version')}
        
        #Update header object with info
        hdr_mask.update(hdr_info_mask)

        #Create fits file with header and data
        
        fits.writeto(fpath_mask, arr_mask, hdr_mask)
        print("Pixel mask fits file saved to %s"%fpath_mask)