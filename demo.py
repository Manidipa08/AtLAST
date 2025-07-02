from atlast import AtLAST
import pickle

atlast_obj = AtLAST(nsample=1, file_path='3c288.fits')

# add bands to the instrument
atlast_obj.add_band(center=93e9, width=53e9, efficiency=0.5, shape="gaussian" ,time_constant=0, gain_error=5e-2, NEP=3e-17, knee=1.00)

# instrument creation
instrument = atlast_obj.create_instrument(primary_size=50, field_of_view=0.25, shape="hexagon", bath_temp=100e-3)

# plot or save
atlast_obj.instrument_plot(save=True)

# scanning pattern
daisy_track = atlast_obj.scanning_pattern(start_time="2022-08-01T23:00:00",
    scan_pattern="daisy",
    scan_options={"radius": 0.05, "speed": 0.02}, # in degrees
    duration=600, # in seconds
    sample_rate=50, # in Hz
    scan_center=(260, -20) # in degrees
)

# plot or save
atlast_obj.scan_pattern_plot(save=True)

# load the input map 
input_map = atlast_obj.load_input_map(resolution=1.00E+00/3.60E+03,center=(260,-20),nu=1.50E+11,units="Jy/pixel")

# plot or save
atlast_obj.input_map_plot(input_map, save=True)

# running the simulation 
tod = atlast_obj.run_sim(site_name="llano_de_chajnantor", atmosphere="2d", atmosphere_kwargs={"weather": {"pwv": 0.5}})

# plot or save
atlast_obj.tod_plot(save=True)

# generate the output map and reproject the input map as per the output map
input_map, output_map, outhdu = atlast_obj.generate_output_map(center=(260,-20),units="uK_RJ",width=0.4,height=0.4,resolution=3.00E+00/3.60E+03,tod_preprocessing={"window": {"name": "tukey", "kwargs": {"alpha": 0.1}},"remove_modes": {"modes_to_remove": [0]},"remove_spline": {"knot_spacing": 10}})

# plot or save
atlast_obj.output_map_plot(output_map, save=True)

# compute the power spectrum
atlast_obj.power_spectrum(input_map, output_map)
atlast_obj.save_PS()


output_file_lists = 'out_power.fits' if atlast_obj.nsample==1 else [f'out_power_{i}.fits' for i in range(atlast_obj.nsample)]


input_file_lists = 'inp_power.fits' if atlast_obj.nsample==1 else [f'inp_power_{i}.fits' for i in range(atlast_obj.nsample)]


# load the power spectrum from the file 
inp_data, outp_data, inp_header, outp_header = atlast_obj.load_PS(input=input_file_lists, output=output_file_lists)


# compute the transfer function
transfer_function, edge_radii, max_radius = atlast_obj.compute_transfer_function(inp_data=inp_data,  outp_data=outp_data)

# plot or save 
atlast_obj.transfer_function_plot(transfer_function, edge_radii,scale='linear' ,save=True)

# compute the beam power spectrum
beam_psd, edge_radii, fwhm_rad, sigma_beam_dimensionless=atlast_obj.detector_beam_PS(max_radius, instrument, atlast_obj.inpsamp, outhdu)

# plot or save
atlast_obj.beam_ps_plot(beam_psd=beam_psd, edge_radii=edge_radii, transfer_function=transfer_function,scale='linear', save=True)

# save the log file ( captures all the events with date and time)
atlast_obj.save_log_file("demo_log_4.txt")

atlast_obj.transfer_function_avg_plot(beam_psd=beam_psd, edge_radii=edge_radii, transfer_function=transfer_function, scale='linear',save=True)