# AtLAST Simulation Library

## Overview
AtLAST (Atacama Large Aperture Submillimeter Telescope) is a simulation library for modeling and analyzing single-dish submillimeter/millimeter observations. The library supports both simulated sky maps and real astronomical data in FITS format, enabling users to study instrument response, scanning strategies, and data reduction pipelines.

## Scientific Motivation
The AtLAST project aims to provide a flexible simulation environment for:
- Testing instrument and scan configurations
- Understanding the impact of atmospheric and instrumental effects
- Quantifying the transfer function and beam response
- Supporting the design and analysis of next-generation submillimeter telescopes

**Calibration Source:**
- The code supports real FITS data, including the well-known radio galaxy 3C 288 (`3c288.fits`), commonly used as a calibration source in radio astronomy.

## Features
- **Simulated and Real Data:** Use either simulated Gaussian sky maps (with unique random seeds per sample) or FITS files as input.
- **Multi-sample Support:** Run parameter studies with multiple samples for robust statistics.
- **Flexible Instrument & Scan Modeling:** Configure bands, instrument geometry, and scan patterns.
- **Atmospheric Modeling:** Choose between 2D and 3D atmospheric models.
- **Comprehensive Plotting:** Generate and save figures for instrument, scan pattern, input/output maps, TOD, power spectra, transfer function, and beam response.
- **Logging & State Saving:** Save logs and simulation state for reproducibility.

## Simulation Workflow
1. **Instrument Setup:**
   - Define bands and instrument geometry.
2. **Input Map:**
   - Use a FITS file (e.g., `3c288.fits`) or generate a simulated Gaussian field (unique seed per sample).
3. **Scan Pattern:**
   - Configure scanning strategy (e.g., daisy pattern).
4. **Run Simulation:**
   - Simulate TOD (Time-Ordered Data) with atmospheric effects.
5. **Map-Making:**
   - Bin TOD into output maps, reproject input maps for comparison.
6. **Power Spectrum & Transfer Function:**
   - Compute and compare input/output power spectra, transfer function, and beam response.
7. **Plotting & Saving:**
   - Save or display all relevant figures and logs.

## Example Usage
```python
from atlast import AtLAST

# Initialize for 3 samples using a FITS file
sim = AtLAST(nsample=3, file_path='3c288.fits')

# Add a band
sim.add_band(center=150, width=30, shape='gaussian', time_constant=0.01, efficiency=0.9, gain_error=0.01, NEP=1e-17, knee=0.1)

# Create instrument
sim.create_instrument(primary_size=6, field_of_view=1, shape='hexagon', bath_temp=0.1)

# Load input map (FITS or simulated)
sim.load_input_map(resolution=0.001, center=[0,0], nu=150, units='uK_RJ')

# Define scan pattern
sim.scanning_pattern(start_time='2025-07-02T00:00:00', scan_pattern='daisy', scan_options={'radius':1}, duration=100, sample_rate=10, scan_center=[0,0])

# Run simulation
sim.run_sim(site_name='atacama', atmosphere='2d', atmosphere_kwargs={})

# Generate output map
sim.generate_output_map(center=[0,0], units='uK_RJ', width=1, height=1, resolution=0.001, tod_preprocessing={})

# Compute power spectra and transfer function
sim.power_spectrum(sim.input_map_data, sim.output_map_data)
sim.save_PS()

inp_data, outp_data, inp_header, outp_header = sim.load_PS('inp_power_0.fits', 'out_power_0.fits')
tf, edge_radii, max_radius = sim.compute_transfer_function(inp_data, outp_data)
beam_psd, edge_radii, fwhm, sigma = sim.detector_beam_PS(max_radius, sim.instrument, inp_data, outp_header)

# Plot results
sim.instrument_plot(save=True)
sim.scan_pattern_plot(save=True)
sim.input_map_plot(sim.input_map_data, save=True)
sim.output_map_plot(sim.output_map_data, save=True)
sim.power_spectrum_plot(sim.outsamp, sim.inpsamp, save=True)
sim.transfer_function_plot(tf, edge_radii, scale='log', save=True)
sim.beam_ps_plot(beam_psd, edge_radii, tf, scale='linear', save=True)
```
  
## Notes
- Simulated sky maps are generated using Gaussian noise with a unique random seed for each sample/run.
- All tests and examples can use the provided `3c288.fits` file for real-data validation.
- The library supports saving all figures and logs for reproducibility.
- All simulation steps are logged internally, and users can save the log file using the `save_log_file` method for later review or reproducibility.

## References
- [AtLAST Project](https://atlast-telescope.org/)
- [Maria Simulation Library](https://github.com/thomaswmorris/maria)
- [Astropy](https://www.astropy.org/)
- [Photutils](https://photutils.readthedocs.io/)
- [van Marrewijk, J., Morris, T. W., Mroczkowski, T., Cicone, C., Dicker, S., Di Mascolo, L., ... & Würzinger, J. (2024). maria: A novel simulator for forecasting (sub-) mm observations. arXiv preprint arXiv:2402.10731.](https://arxiv.org/abs/2402.10731)

---
For questions or contributions, please open an issue or pull request on GitHub.
