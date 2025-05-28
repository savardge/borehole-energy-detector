# borehole-energy-detector

Detects events using energy ratio methods on borehole 3C geophone waveform data. Uses an STA-LTA coincidence trigger, then picks using the Joint Energy Ratio method (Akram et al., 2013), refine picks with multi-channel cross-correlation (VanDecar and Crosson, 1990), and picks the magnitude for S-wave arrivals using both a time domain and spectral domain approaches (Stork et al., 2014). Bash scripts for job array SLURM scheduling included.

Work in progress, not intended for distribution yet!
