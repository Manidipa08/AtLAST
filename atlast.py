import maria
from maria import Array
from maria import Instrument
import matplotlib.pyplot as plt
from maria.mappers import BinMapper
from maria.instrument import Band
from maria import map
from scipy.integrate import trapezoid
import numpy as np
from astropy.io import fits
import reproject
from reproject import reproject_interp
from astropy import units as u
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.modeling.models import Gaussian2D
from photutils.centroids import *
from photutils.datasets import make_noise_image
from photutils.profiles import RadialProfile
from datetime import datetime
import pickle
import os
import random

class AtLAST:
    def __init__(self, nsample: int = 1, file_path: str = None):
        self.nsample = nsample
        self.inpsamp = None
        self.outpsamp = None

        self.array_data = None
        self.input_map_data = None
        self.output_map_data = None
        self.instrument = None
        self.track = None
        self.tod = None
        self.bands = []
        self.units = 'uK_RJ'
        self.frame = 'ra_dec'
        self.allowed_units = ['uK_RJ','rad','deg','arcmin','arcsec','s','min','hour','day','week','year','g','m','Hz','W√s','K√s','K_RJ','F_RJ','K_b','K_CMB','W','Jy','Jy/pixel']
        self.allowed_shapes = ['hexagon','circular']
        self.allowed_scan_patterns = ['daisy', 'double_circle', 'lissajous',  'back_and_forth', 'stare']
        self.allowed_band_shape = ['gaussian']
        self.allowed_atmospheres = ['2d','3d']
        self._logger = ""
        self.seed_value = None
        self.input_file_path = None
        self.radial_new_stuff = None
        self.fwhm_rad_inv = None 
        self.fov_rad_inv = None 
        if file_path is not None:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"The file {file_path} does not exist.")
            if self.nsample == 1:
                self.input_file_path = file_path
            else:
                self.input_file_path = []
                for i in range(nsample):
                    self.input_file_path.append(file_path)


    def get_logger(self):
        print(self._logger)

    def _log_data(self, data):
        self._logger+=(str(datetime.now()) + ":  " + data + "\n")
    
    def save_log_file(self, file_name):
        with open(file_name, 'w') as f:
            f.write("ATLAST Simulation Log START\n")
            f.write("Date: " + str(datetime.now()) + "\n")
            f.write("====================================\n")
            f.write(self._logger)
            f.write("====================================\n")
            f.write("ATLAST Simulation Log END\n")
    

    def add_band(self, center, width, shape, time_constant, efficiency, gain_error, NEP, knee):
        """
        
        """
        if shape not in self.allowed_band_shape:
            raise ValueError(f"Invalid band shape. Allowed shapes are: {self.allowed_band_shape}")
        band = Band(center=center, width=width, shape=shape, time_constant=time_constant, efficiency=efficiency, gain_error=gain_error, NEP=NEP, knee=knee)
        self.bands.append(band)
        self._log_data(f"Band added: {band}")
        print("Band added")

        return band
    
    def create_instrument(self, primary_size, field_of_view, shape, bath_temp, n=None):
        '''

        :param primary_size: size of the single dish in meters
        :param field_of_view: the focal plane in degrees
        :param shape: ['hexagon', 'circular']
        :param bath_temp:
        :param n:
        :return: instrument object
        '''
        if self.bands == []:
            raise ValueError("No bands added to the instrument. Please add bands using the add_band method.")
        array= {"primary_size": primary_size,
                "field_of_view": field_of_view,
                "shape": shape,
                "bath_temp": bath_temp,
                "n": n, # numbers of detectors
                "bands": self.bands,
                }
        self.fov_rad_inv = (1/np.deg2rad(field_of_view))        
        if shape not in self.allowed_shapes:
            raise ValueError(f"Invalid shape. Allowed shapes are: {self.allowed_shapes}")
        
        self.array_data = array
        instrument = maria.get_instrument(array=self.array_data)
        self.instrument = instrument
        self._log_data(f"Instrument created: {instrument}")
        print(instrument)
        return instrument
    
    
    def fwhm_instrument(self, instrument):
        fwhm = instrument.arrays.angular_fwhm(z=np.inf).mean()
        fwhm = fwhm*u.rad
        self._log_data(f"FWHM of the instrument: {fwhm}")
        return fwhm #in radian 
    

    def scanning_pattern(self, start_time, scan_pattern, scan_options, duration, sample_rate, scan_center):
        if scan_pattern not in self.allowed_scan_patterns:
            raise ValueError(f"Invalid scan pattern. Allowed scan patterns are: {self.allowed_scan_patterns}")
        if scan_pattern == 'daisy':
            self.track = maria.get_plan(start_time=start_time,
                                scan_pattern=scan_pattern,
                                scan_options=scan_options, # in degrees
                                duration=duration, # in seconds
                                sample_rate=sample_rate, # in Hz
                                frame=self.frame,
                                scan_center= scan_center # in degrees
                                )#time taken as mentioned the configuration in Maria paper
        elif scan_pattern == 'double_circle':
            self.track = maria.get_plan(start_time=start_time,
                                scan_pattern=scan_pattern,
                                scan_options=scan_options, # e.g., {'radius': 2, 'speed': 0.25, 'miss_freq': 10.1}
                                duration=duration, # in seconds
                                sample_rate=sample_rate, # in Hz
                                frame=self.frame,
                                scan_center= scan_center # in degrees
                                )
        elif scan_pattern == 'lissajous':
            self.track = maria.get_plan(start_time=start_time,
                                scan_pattern=scan_pattern,
                                scan_options=scan_options, # e.g., {'radius, speed}
                                duration=duration, # in seconds
                                sample_rate=sample_rate, # in Hz
                                frame=self.frame,
                                scan_center= scan_center # in degrees
                                )


        elif scan_pattern == 'back_and_forth':
            self.track = maria.get_plan(start_time=start_time,
                                scan_pattern=scan_pattern,
                                scan_options=scan_options, # e.g., {'radius': 5, 'speed': 0.5}
                                duration=duration, # in seconds
                                sample_rate=sample_rate, # in Hz
                                frame=self.frame,
                                scan_center= scan_center # in degrees
                                )
        elif scan_pattern == 'stare':
            self.track = maria.get_plan(start_time=start_time,
                                scan_pattern=scan_pattern,
                                scan_options=scan_options, # e.g., {'amplitude': 1, 'frequency': 1};{}
                                duration=duration, # in seconds
                                sample_rate=sample_rate, # in Hz
                                frame=self.frame,
                                scan_center= scan_center # in degrees
                                )
        self._log_data(f"Scanning pattern created: {self.track}")
        print(self.track)
        return self.track
    
    def gen_gaussian_field(self, nx, ny, gamma=-2.5, seed=None):
        if seed is not None:
            np.random.seed(54)
        
        x = np.fft.fftfreq(nx).reshape(nx, 1)
        y = np.fft.fftfreq(ny).reshape(1, ny)
        k_squared = x**2 + y**2 #in fourier space
        k_squared[0, 0] = 1.0  # Avoid division by zero at the origin

        # P(k) ~ k^gamma
        ps = k_squared ** (gamma)

        np.random.seed(seed)
        random_phases = np.exp(2j * np.pi * np.random.rand(nx, ny))
        fourier_field = random_phases * np.sqrt(ps)


        field = np.fft.ifft2(fourier_field).real

        # Normalize the field
        field -= field.mean()
        field /= field.std()

        return field
    

    def load_input_map(self, resolution, center, nu, units: str):
        """
            params: 
                :param resolution: pixel size in degrees
                :param center: position in the sky
                :param nu: frequency of the map
                :param units: Unit of the input map

            :return: input_map
        """
        if units not in self.allowed_units:
            raise ValueError(f"Invalid units. Allowed units are: {self.allowed_units}")
        if self.nsample==1:
            if self.input_file_path is None:
                metadata = {
                        "resolution" : resolution,
                        "center" : center,
                        "nu" : nu,
                        "units" : units
                        }
                field = self.gen_gaussian_field(6000, 6000,-1.0 )
                # white noise
                # input_map = maria.map.ProjectedMap(data=1.00E-03*np.random.rand(6000,6000),degrees=True,**metadata)
                input_map = maria.map.ProjectedMap(data=field,degrees=True,**metadata)
            else:
                input_map = maria.map.load(
                    nu=nu,
                    filename=self.input_file_path,
                    resolution= resolution,  # pixel size in degrees
                    center=center,  # position in the sky
                    units=units,  # Units of the input map
                )
        else:
            input_map = []
            self.seed_value = []
            if self.input_file_path is None:
                for i in range(self.nsample):
                    metadata = {
                        "resolution" : resolution,
                        "center" : center,
                        "nu" : nu,
                        "units" : units
                        }
                    seed_value = random.randint(0,(2**32)-1)
                    # np.random.seed(seed_value)
                    field = self.gen_gaussian_field(6000, 6000,-1.0 , seed_value)
                    # input_map.append(maria.map.ProjectedMap(data=1.00E-03*np.random.rand(6000,6000),degrees=True,**metadata))
                    input_map.append(maria.map.ProjectedMap(data=field,degrees=True,**metadata))
                    self.seed_value.append(seed_value)
            else:
                if len(self.input_file_path)==self.nsample:
                    for i in range(self.nsample):
                        input_map.append(maria.map.load(
                        nu=nu,
                        filename=self.input_file_path[i],
                        resolution= resolution,  # pixel size in degrees
                        center=center,  # position in the sky
                        units=units,  # Units of the input map
                    ))
                else:
                    raise ValueError("incorrect length of input_files")
        print(input_map)
        self._log_data(f"Input map loaded: {input_map}")
        self.input_map_data = input_map
        return input_map
    


    def run_sim(self, site_name: str , atmosphere, atmosphere_kwargs):
        if atmosphere not in self.allowed_atmospheres:
            raise ValueError(f"Invalid atmosphere. Allowed atmospheres are: {self.allowed_atmospheres}")
        site = maria.get_site(site_name)
        print(site)
        print("Simulation started ")
        self._log_data(f"Simulation started with site: {site}")
        if self.nsample==1:
            sim = maria.Simulation(
                        instrument = self.instrument, 
                        map=self.input_map_data,
                        site = site,
                        plan = self.track,
                        atmosphere = atmosphere,
                        atmosphere_kwargs= atmosphere_kwargs#,"seed": 12445},
                        )
            print(sim)
            tod = sim.run()
            self.tod = tod
            self._log_data(f"Simulation computed: {sim}")
            pickle.dump(tod, open("tod_data.pkl", "wb"))
            return tod
        else:
            self.tod = []
            for i in range(self.nsample):
                print(f"simulation started for {i} sample")
                sim = maria.Simulation(
                        instrument = self.instrument, 
                        map=self.input_map_data[i],
                        site = site,
                        plan = self.track,
                        atmosphere = atmosphere,
                        atmosphere_kwargs= atmosphere_kwargs#,"seed": 12445},
                        )
                print(sim)
                tod_val= sim.run()
                self.tod.append(tod_val)
                self._log_data(f"Simulation computed for {i} sample: {sim}")
                # pickle.dump(tod_val, open(f"tod_data_{i}.pkl", "wb"))
            return self.tod # list
                
    

    def load_tod_from_file(self, file_path):
        """
        Load the Time-Ordered Data (TOD) from a pickle file.
        The file is generated by the run_sim method.
        This method is useful for loading precomputed TOD data without running the simulation again.
    
        :param file_path: Path to the pickle file containing the TOD.
        :return: Loaded TOD object.
        """
        if self.nsample == 1:
            with open(file_path, 'rb') as f:
                self.tod = pickle.load(f)
            print("TOD loaded from file")
            self._log_data(f"TOD loaded from file: {self.tod}")
            return self.tod
        else:
            self.tod = []
            for i in range(self.nsample):
                with open(file_path, 'rb') as f:
                    self.tod.append(pickle.load(f))
                print(f"TOD loaded from file for sample {i}")
                self._log_data(f"TOD loaded from file: {self.tod[i]}")


    def generate_output_map(self, center, units, width, height, resolution, tod_preprocessing, map_postprocessing=None):
        if units not in self.allowed_units:
            raise ValueError(f"Invalid units. Allowed units are: {self.allowed_units}")
        mapper = BinMapper(
                    center = center,
                    frame = self.frame,
                    units = units,
                    width = width, #4*0.25,
                    height = height, #4*0.25,
                    resolution = resolution, #3.00E+00/3.60E+03
                    tod_preprocessing = tod_preprocessing #{"window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
                    #                                 "remove_modes": {"modes_to_remove": [0]},
                    #                                 "remove_spline": {"knot_spacing": 10}},
                    #map_postprocessing = {"median_filter": 0}#{"size": 1}}#"gaussian_filter": {"sigma": 1},
                )
        
        if self.nsample==1:
            mapper.add_tods(self.tod)
            self.output_map_data = mapper.run()
            self.output_map_data.data[np.isnan(self.output_map_data.data)] = 0.00

            print(self.output_map_data)
            self._log_data(f"Output map generated: {self.output_map_data}")
            print("Output map generated")

            outhdu = fits.PrimaryHDU(self.output_map_data.data[0,0,0],header=self.output_map_data.header)
            self.input_map_data,_ = reproject.reproject_interp((self.input_map_data.to(units=self.units).data[0,0,0],self.input_map_data.header),outhdu.header)#reprojecting into output scale
            self.input_map_data[np.isnan(self.input_map_data)] = 0.00
            print(np.shape(self.input_map_data))
            print("Reprojecting input map into output map")
            self._log_data(f"Reprojected input map: {self.input_map_data}")
            return  self.input_map_data,self.output_map_data, outhdu 
        else:
            self.output_map_data = []
            outhdu = []
            for i in range(self.nsample):
                mapper = BinMapper(
                    center = center,
                    frame = self.frame,
                    units = units,
                    width = width, #4*0.25,
                    height = height, #4*0.25,
                    resolution = resolution, #3.00E+00/3.60E+03
                    tod_preprocessing = tod_preprocessing #{"window": {"name": "tukey", "kwargs": {"alpha": 0.1}},
                    #                                 "remove_modes": {"modes_to_remove": [0]},
                    #                                 "remove_spline": {"knot_spacing": 10}},
                    #map_postprocessing = {"median_filter": 0}#{"size": 1}}#"gaussian_filter": {"sigma": 1},
                )
                mapper.add_tods(self.tod[i])
                self.output_map_data.append(mapper.run())
                self.output_map_data[i].data[np.isnan(self.output_map_data[i].data)] = 0.00

                print(self.output_map_data[i])
                self._log_data(f"Output map generated: {self.output_map_data[i]}")
                print("Output map generated")

                outhdu.append(fits.PrimaryHDU(self.output_map_data[i].data[0,0,0],header=self.output_map_data[i].header))

                self.input_map_data[i],_ = reproject.reproject_interp((self.input_map_data[i].to(units=self.units).data[0,0,0],self.input_map_data[i].header),outhdu[0].header)#reprojecting into output scale
                self.input_map_data[i][np.isnan(self.input_map_data[i])] = 0.00
                print(np.shape(self.input_map_data[i]))
                print("Reprojecting input map into output map")
                self._log_data(f"Reprojected input map: {self.input_map_data[i]}")
            return  self.input_map_data,self.output_map_data, outhdu ## output_map_data and outhdu is list  
                
        



    # def reproject_input_map(self, input_map, output_map):
    #     outhdu = fits.PrimaryHDU(output_map.data[0,0,0],header=output_map.header)
    #     input_map,_ = reproject.reproject_interp((input_map.to(units='uK_RJ').data[0,0,0],input_map.header),outhdu.header)#reprojecting into output scale
    #     print(np.shape(input_map))
    #     return input_map, outhdu
    
    def power_spectrum(self, input_map, output_map):

        

        if self.nsample==1:
            self.inpsamp = np.fft.fftshift(np.fft.fft2(np.squeeze(input_map.data),axes=(-2,-1)),axes=(-2,-1))
            self.inpsamp = np.abs(self.inpsamp)**2
            self._log_data(f"Power spectrum computed (input sample): {self.inpsamp}")
            output_map = output_map.to(units=self.units)
            self.outsamp = np.fft.fftshift(np.fft.fft2(np.squeeze(output_map.data),axes=(-2,-1)),axes=(-2,-1))
            self.outsamp = np.abs(self.outsamp)**2
            # outsamp = np.mean(outsamp, axis=0)
            # print(np.shape(self.outsamp))
            # inpsamp = np.mean(inpsamp,axis=0)
            # print(np.shape(self.inpsamp))
            self._log_data(f"Power spectrum computed (output sample): {self.outsamp}")
            
        else:
            self.outsamp = []
            self.inpsamp = []
            for i in range(self.nsample):

                self.inpsamp.append(np.abs(np.fft.fftshift(np.fft.fft2(np.squeeze(input_map[i].data),axes=(-2,-1)),axes=(-2,-1)))**2)
                # self.inpsamp[i] = (np.abs(self.inpsamp[i])**2)
                self._log_data(f"Power spectrum computed (input sample): {self.inpsamp[i]}")

                output_map[i]  = output_map[i].to(units=self.units)
                self.outsamp.append(np.abs(np.fft.fftshift(np.fft.fft2(np.squeeze(output_map[i].data),axes=(-2,-1)),axes=(-2,-1)))**2)
                self._log_data(f"Power spectrum computed (output sample): {self.outsamp[i]}")


    
    
    def save_PS(self):

        
        if self.nsample==1:
            fits.writeto('inp_power.fits',data=self.inpsamp,overwrite=True)
            fits.writeto('out_power.fits',data=self.outsamp,overwrite=True)
            
            self._log_data("Power spectrum saved")
        else:
            for i in range(self.nsample):
                fits.writeto(f'inp_power_{i}.fits',data=self.inpsamp[i],overwrite=True)
                fits.writeto(f'out_power_{i}.fits',data=self.outsamp[i],overwrite=True)
                self._log_data(f"Power spectrum saved for {i} sample")
        print("Power spectrum saved")


    def load_PS(self, input: str, output):
        '''
        params: 
            :param inp: input power spectrum file path
            :param outp: output power spectrum file path
        :return: inp_data, outp_data, inp_header, outp_header
        '''
        

        if self.nsample==1:
            input = fits.open(input)
            inp_data = input[0].data
            inp_header = input[0].header
            self._log_data(f"Power spectrum loaded (input): {inp_data}")

            output = fits.open(output)
            outp_data = output[0].data
            outp_header = output[0].header
            
            self._log_data(f"Power spectrum loaded (output): {outp_data}")
        else:
            outp_data = []
            outp_header = []
            inp_data = []
            inp_header = []
            for i in range(self.nsample):
                input[i] = fits.open(input[i])
                inp_data.append(input[i][0].data)
                inp_header.append(input[i][0].header)
                self._log_data(f"Power spectrum loaded (input): {inp_data}")

                output[i] = fits.open(output[i])
                outp_data.append(output[i][0].data)
                outp_header.append(output[i][0].header)
                self._log_data(f"Power spectrum loaded for {i} sample (output): {outp_data[i]}")
        return inp_data, outp_data, inp_header, outp_header
    
    def compute_transfer_function(self, inp_data, outp_data):
        # Compute radial profiles
        # ny, nx = inp_data.shape
        # center = (float(ny // 2), float(nx // 2))
        # print(center)
        # print(min(inp_data[0].shape))
        # max_radius = min(inp_data.shape) // 2

        # edge_radii=np.linspace(0,max_radius,100*max_radius)# creating finer sample of the radial profiles; no. of the radial bins
        # # edge_radii = np.arange(0, max_radius + 1)

        
        if self.nsample==1:
            ny, nx = inp_data.shape
            center = (float(ny // 2), float(nx // 2))
            print(center)
            print(min(inp_data.shape))
            max_radius = min(inp_data.shape) // 2

            edge_radii=np.linspace(0,max_radius,100*max_radius)# creating finer sample of the radial profiles; no. of the radial bins
            # edge_radii = np.arange(0, max_radius + 1)

            radial_inp = RadialProfile(inp_data, center, edge_radii)
            radial_outp = RadialProfile(outp_data, center, edge_radii)
            print("Transfer function computation started")
            self._log_data(f"Transfer function computation started")    
            min_len = min(len(radial_inp.profile), len(radial_outp.profile))
            profile_inp = radial_inp.profile[:min_len]
            profile_outp = radial_outp.profile[:min_len]
            radii = radial_inp.radius[:min_len]
            self.radial_new_stuff = radial_outp.profile 
            # Compute transfer function
            # eps = 1e-10  # To avoid divide-by-zero
            print("Computing transfer function sqrt")
            transfer_function = np.sqrt(profile_outp/ (profile_inp))
            print("Computing transfer function done")
            # transfer_function = (transfer_function/np.max(transfer_function))
            edge_radii = edge_radii[0:len(transfer_function)]
            self._log_data(f"Transfer function computed: {transfer_function}")
            print("Transfer function computed")
            return transfer_function, edge_radii, max_radius
        
        else:
            

            radial_outp = []
            min_len = []
            profile_outp = []
            radii = []
            transfer_function = []
            radial_inp = []
            profile_inp = []
            # edge_radii = []
            for i in range(self.nsample):
                ny, nx = inp_data[i].shape
                center = (float(ny // 2), float(nx // 2))
                print(center)
                print(min(inp_data[i].shape))
                max_radius = min(inp_data[i].shape) // 2

                edge_radii=np.linspace(0,max_radius,100*max_radius)# creating finer sample of the radial profiles; no. of the radial bins
        # edge_radii = np.arange(0, max_radius + 1)
                radial_inp.append(RadialProfile(inp_data[i], center, edge_radii))
                radial_outp.append(RadialProfile(outp_data[i], center, edge_radii))
                print(f"Transfer function computation started for {i} sample")
                self._log_data(f"Transfer function computation started for {i} sample")    
                min_len.append(min(len(radial_inp[i].profile), len(radial_outp[i].profile)))
                profile_inp.append(radial_inp[i].profile[:min_len[i]])
                profile_outp.append(radial_outp[i].profile[:min_len[i]])
                radii.append(radial_inp[i].radius[:min_len[i]])

                # Compute transfer function
                # eps = 1e-10  # To avoid divide-by-zero
                transfer_function.append(np.sqrt(profile_outp[i]/ (profile_inp[i])))
                # transfer_function[i] = (transfer_function[i]/np.max(transfer_function[i]))
                edge_radii = edge_radii[0:len(transfer_function[i])]
                self.radial_new_stuff = radial_outp[i].profile
                self._log_data(f"Transfer function computed for {i} sample: {transfer_function}")
            print("Transfer function computed")
            return transfer_function, edge_radii, max_radius            
    

    def detector_beam_PS(self, max_radius, instrument, inp_data, outhdu):

        '''
        to discuss here about nx_beam and ny_beam
        '''
        fwhm = self.fwhm_instrument(instrument)
        edge_radii=np.linspace(0,max_radius,100*max_radius)#radial frequency bins; radians/pixel; k = no. of rad/lambda in definition, k*lambda = no. of rad/pixels
        fwhm_rad = fwhm
        self.fwhm_rad_inv = 1/fwhm_rad #in radians^-1
        if self.nsample==1:
            # sigma_beam = fwhm_rad / np.sqrt(8 * np.log(2)) #in radians
            scale = fwhm.to(u.deg).value/np.sqrt(8.00*np.log(2.00))/outhdu.header['CDELT2']
            print(scale)
            nx_beam, ny_beam = inp_data.shape
            deg = outhdu.header['CDELT2']
            rad = np.deg2rad(deg)
            # edge_radii_safe = np.copy(radial_profile_beam.radius)
            # edge_radii_safe[edge_radii_safe==0] = np.nan
            # spat_freq = ((radial_profile_beam.radius*2)/(rad*(nx_beam)))

        else:
            scale = fwhm.to(u.deg).value/np.sqrt(8.00*np.log(2.00))/outhdu[0].header['CDELT2']
            print(scale)
            nx_beam, ny_beam = inp_data[0].shape
            deg = outhdu[0].header['CDELT2']
            rad = np.deg2rad(deg)
            # edge_radii_safe = np.copy(radial_profile_beam.radius)
            # edge_radii_safe[edge_radii_safe==0] = np.nan
            # spat_freq = ((radial_profile_beam.radius*2)/(rad*(nx_beam)))
        # Beam transfer function
        sigma_beam_dimensionless = scale
        beam_size = int(sigma_beam_dimensionless)  # Convert to pixels
          # Use input data shape
        print(nx_beam)
        x_beam, y_beam = np.meshgrid(np.arange(nx_beam), np.arange(ny_beam))
        print("Beam transfer function computation started")
        self._log_data(f"Beam transfer function computation started")
        gaussian_beam_model = Gaussian2D(
            amplitude=1,
            x_mean=nx_beam / 2,
            y_mean=ny_beam / 2,
            x_stddev=beam_size,
            y_stddev=beam_size,
            theta=0  # Assuming no rotation
        )
        gaussian_beam_2D = gaussian_beam_model(x_beam, y_beam)
        beam_psd_2D = np.abs(np.fft.fftshift(np.fft.fft2(gaussian_beam_2D)))
    
        radial_profile_beam = RadialProfile(beam_psd_2D, (nx_beam/2, ny_beam/2), edge_radii) 
        beam_psd = radial_profile_beam.profile / max(radial_profile_beam.profile)
        spatial_freq = ((radial_profile_beam.radius*2)/(rad*(nx_beam)))

        # spatial_freq = (edge_radii_safe / (rad*len(self.radial_new_stuff)))
        self._log_data(f"Beam power spectrum computed: {beam_psd}")
        print("Beam transfer function computation completed")
        # gaussian_beam_2D = np.exp(-((x_beam - nx_beam/(2))**2
        return beam_psd, spatial_freq, fwhm_rad, sigma_beam_dimensionless
    
        

    #########################PLOTTING FUNCTIONS###########################

    def instrument_plot(self, save: bool = False):
        self.instrument.plot()
        plt.title("Instrument")
        if save:
            plt.savefig("instrument.png")   
            plt.close()
        else:   
            plt.show()
    

    def scan_pattern_plot(self, save: bool = False):
        self.track.plot()
        plt.title("Scanning Pattern")
        if save:    
            plt.savefig("scan_pattern.png")
            plt.close()
        else:
            plt.show()

    # def input_map_plot(self, input_map, save: bool = False):
    #     if self.nsample==1:
    #         input_map.to(units=self.units).plot()
    #         plt.title("Input Map")
    #         if save:
    #             plt.savefig("input_map.png")
    #             plt.close()
    #         else:
    #             plt.show()
    #     else:
    #         for i in range(self.nsample):
    #             input_map[i].to(units=self.units).plot()
    #             plt.title("Input Map")
    #             if save:
    #                 plt.savefig(f"input_map_{i}.png")
    #                 plt.close()
    #             else:
    #                 plt.show()
    def input_map_plot(self, input_map, save: bool = False):
        if self.nsample == 1:
            if isinstance(input_map, np.ndarray):
                plt.imshow(input_map)
                plt.title("Reprojected Input Map")
                plt.colorbar()
                if save:
                    plt.savefig("reprojected_input_map.png")
                    plt.close()
                else:
                    plt.show()
            else:
                input_map.to(units=self.units).plot()
                plt.title("Input Map")
                if save:
                    plt.savefig("input_map.png")
                    plt.close()
                else:
                    plt.show()
        else:
            for i in range(self.nsample):
                if isinstance(input_map[i], np.ndarray):
                    plt.imshow(input_map[i])
                    plt.title("Reprojected Input Map")
                    plt.colorbar()
                    if save:
                        plt.savefig(f"reprojected_input_map_{i}.png")
                        plt.close()
                    else:
                        plt.show()
                else:
                    input_map[i].to(units=self.units).plot()
                    plt.title("Input Map")
                    if save:
                        plt.savefig(f"input_map_{i}.png")
                        plt.close()
                    else:
                        plt.show()


    def tod_plot(self, save: bool = False):
        if self.nsample==1:
            self.tod.plot()
            plt.title("Time-Ordered Data (TOD)")
            if save:
                plt.savefig("tod.png")
                plt.close()
            else:
                plt.show()
        else:
            for i in range(self.nsample):
                self.tod[i].plot()
                plt.title(f"Time-Ordered Data (TOD): {i}")
                if save:
                    plt.savefig(f"tod_{i}.png")
                    plt.close()
                else:
                    plt.show()

    def output_map_plot(self, output_map, save: bool = False):
        if self.nsample == 1:
            output_map.to(units=self.units).plot()
            plt.title("Output Map")
            if save:
                plt.savefig("output_map.png")
                plt.close()
            else:
                plt.show()
        else:
            for i in range(self.nsample):
                output_map[i].to(units=self.units).plot()
                plt.title(f"Output Map: {i}")
                if save:
                    plt.savefig(f"output_map_{i}.png")
                    plt.close()
                else:
                    plt.show()


    def power_spectrum_plot(self, outsamp, inpsamp, save: bool = False):
        if self.nsample == 1:
            plt.title("PS of output")
            plt.imshow(outsamp)
            if save:
                plt.savefig("output_power_spectrum.png")
                plt.close()
            else:
                plt.show()
            plt.title("PS of reprojected input")
            plt.imshow(inpsamp)
            if save:
                plt.savefig("input_power_spectrum.png")
                plt.close()
            else:
                plt.show()
        else:
            for i in range(self.nsample):
                plt.title(f"PS of output: {i}")
                plt.imshow(outsamp[i])
                if save:
                    plt.savefig(f"output_power_spectrum_{i}.png")
                    plt.close()
                else:
                    plt.show()
                plt.title(f"PS of reprojected input: {i}")
                plt.imshow(inpsamp[i])
                if save:
                    plt.savefig(f"input_power_spectrum_{i}.png")
                    plt.close()
                else:
                    plt.show()


    def transfer_function_plot(self, transfer_function, edge_radii, scale= ('linear', 'linear') ,save: bool = False, file_name = 'transfer_function.png'):
        if self.nsample == 1:
            plt.figure(figsize=(8, 5))
            plt.plot(edge_radii, transfer_function, label='Transfer Function')
            plt.xscale(scale[0])
            plt.yscale(scale[1])
            plt.xlabel("Spatial Frequency")
            plt.ylabel("T(k)")
            plt.title("Transfer Function from Precomputed Power Spectra")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(file_name)
                plt.close()  # Close the figure after saving
            else:
                plt.show()
        else:
            plt.figure(figsize=(8, 5))
            for i in range(self.nsample):
                min_len = min(len(edge_radii), len(transfer_function[i]))
                plt.plot(edge_radii[:min_len], transfer_function[i][:min_len], label=f'Transfer Function {i}')
            plt.xscale(scale[0])
            plt.yscale(scale[1])
            plt.xlabel("Spatial Frequency")
            plt.ylabel("T(k)")
            plt.title("Transfer Function from Precomputed Power Spectra")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig("transfer_function_with_all_samples.png")
                plt.close()  # Close the figure after saving
            else:
                plt.show()

    def beam_ps_plot(self, beam_psd, spatial_freq, transfer_function, scale = ('linear', 'linear') ,save: bool = False, file_name = 'transfer_function_and_beam_ps.png'):
        if self.nsample == 1:
            plt.figure(figsize=(8, 5))  # Create new figure
            plt.plot(spatial_freq, beam_psd,  label='Beam Power Spectrum', linestyle='--')
            if self.fwhm_rad_inv is None and self.fov_rad_inv is None:
                raise ValueError("fwhm_rad_inv and fov_rad_inv must be set before plotting")
            
            plt.axhline(0.5, color='grey', linestyle='--', label='FWHM=0.5')
            plt.axvline(1.22e4, color='black', linestyle='--', label='FWHM')
            plt.axvline(229.2, color='purple', linestyle='--', label='FoV')
            plt.plot(spatial_freq, transfer_function, label='Transfer Function')
            plt.xscale(scale[0])
            plt.yscale(scale[1])
            plt.xlim(1e1,max(spatial_freq))
            plt.xlabel("Spatial Frequency (in rad$^{-1}$ unit)")
            plt.ylabel("T(k)")
            plt.title("Transfer Function and Beam Power Spectrum")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(file_name)
                plt.close()  # Close the figure after saving
            else:
                plt.show()
        else:
            plt.figure(figsize=(8, 5))  # Create new figure
            plt.plot(spatial_freq, beam_psd,  label='Beam Power Spectrum', linestyle='--')
            if self.fwhm_rad_inv is None and self.fov_rad_inv is None:
                raise ValueError("fwhm and field of view must be set before plotting")
            plt.axhline(0.5, color='grey', linestyle='--', label='FWHM=0.5')
            plt.axvline(1.22e4, color='black', linestyle='--', label='FWHM')
            plt.axvline(229.2, color='purple', linestyle='--', label='FoV')
            for i in range(self.nsample):
                min_len = min(len(spatial_freq), len(transfer_function[i]))
                plt.plot(spatial_freq[:min_len], transfer_function[i][:min_len], label=f'Transfer Function {i}')
            plt.xscale(scale[0])
            plt.yscale(scale[1])
            plt.xlim(1e1,max(spatial_freq))
            plt.xlabel("Spatial Frequency (in rad$^{-1}$ unit)")
            plt.ylabel("T(k)")
            plt.title(f"Transfer Function and Beam Power Spectrum for all samples")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(f"transfer_function_and_beam_ps_all_samples.png")
                plt.close()  # Close the figure after saving
            else:
                plt.show()

    def transfer_function_avg_plot(self, beam_psd, spatial_freq, transfer_function, scale = ('linear', 'linear') ,save: bool = False):
        if self.nsample == 1:
            print("Error: average cant be calculated for single sample")
        else:
            plt.figure(figsize=(8, 5))  # Create new figure
            plt.plot(spatial_freq, beam_psd,  label='Beam Power Spectrum', linestyle='--')
            if self.fwhm_rad_inv is None and self.fov_rad_inv is None:
                raise ValueError("fwhm_rad_inv and fov_rad_inv must be set before plotting")
            plt.axhline(0.5, color='grey', linestyle='--', label='FWHM=0.5')
            plt.axvline(1.22e4, color='black', linestyle='--', label='FWHM')
            plt.axvline(229.2, color='purple', linestyle='--', label='FoV')
            min_tf_len = min(len(tf) for tf in transfer_function)
            min_len = min(len(spatial_freq), min_tf_len)
            updated_tf = [tf[:min_len] for tf in transfer_function]
            avg_tf = np.mean(updated_tf, axis=0)
            plt.plot(spatial_freq[:min_len], avg_tf[:min_len], label=f'Average Transfer Function')
            plt.xscale(scale[0])
            plt.yscale(scale[1])
            plt.xlim(1e1,max(spatial_freq))
            plt.xlabel("Spatial Frequency (in rad$^{-1}$ unit)")
            plt.ylabel("T(k)")
            plt.title(f"Transfer Function Average and Beam Power Spectrum ")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            if save:
                plt.savefig(f"transfer_function_avg_and_beam_ps.png")
                plt.close()  # Close the figure after saving
            else:
                plt.show()


    def save_state(self, filename: str = "atlast_state.pkl"):
        """
        Save all class variables to a pickle file.
        
        Args:
            filename (str): Name of the pickle file to save the state
        """
        state_dict = {
            'nsample': self.nsample,
            'inpsamp': self.inpsamp,
            'outpsamp': self.outpsamp,
            'array_data': self.array_data,
            'input_map_data': self.input_map_data,
            'output_map_data': self.output_map_data,
            'instrument': self.instrument,
            'track': self.track,
            'tod': self.tod,
            'bands': self.bands,
            'units': self.units,
            'frame': self.frame,
            '_logger': self._logger
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(state_dict, f)
        print(f"State saved to {filename}")
        self._log_data(f"State saved to {filename}")

    def load_state(self, filename: str = "atlast_state.pkl"):
        """
        Load all class variables from a pickle file.
        
        Args:
            filename (str): Name of the pickle file to load the state from
        """
        try:
            with open(filename, 'rb') as f:
                state_dict = pickle.load(f)
            
            # Restore all variables
            self.nsample = state_dict['nsample']
            self.inpsamp = state_dict['inpsamp']
            self.outpsamp = state_dict['outpsamp']
            self.array_data = state_dict['array_data']
            self.input_map_data = state_dict['input_map_data']
            self.output_map_data = state_dict['output_map_data']
            self.instrument = state_dict['instrument']
            self.track = state_dict['track']
            self.tod = state_dict['tod']
            self.bands = state_dict['bands']
            self.units = state_dict['units']
            self.frame = state_dict['frame']
            self._logger = state_dict['_logger']
            
            print(f"State loaded from {filename}")
            self._log_data(f"State loaded from {filename}")
            
        except FileNotFoundError:
            print(f"Error: File {filename} not found")
            self._log_data(f"Error: File {filename} not found")
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            self._log_data(f"Error loading state: {str(e)}")


