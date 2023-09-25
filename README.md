# licsar_extra
Tools for advanced processing of LiCSAR data (starting from geotiffs).
Primarily for unwrapping interferograms, primarily produced by LiCSAR system, integrated to LiCSBAS

You may install this set of tools e.g. using pip by:
pip install git+https://github.com/comet-licsar/licsar_extra.git

Then, to test it try e.g.:
`from licsar_extra import lics_unwrap
help(lics_unwrap.process_ifg)`