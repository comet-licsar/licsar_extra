from setuptools import setup

setup(
    name='licsar_extra',
    version='0.2',
    description='LiCSAR Extra set of tools',
    url='https://github.com/comet-licsar/licsar_extra',
    author='Milan Lazecky',
    author_email='M.Lazecky@leeds.ac.uk',
    license='GNU-GPL3.0',
    packages=['licsar_extra'],
    package_dir = {'licsar_extra':'python'},
    python_requires='>=3.9',
    install_requires = ['numpy',
'matplotlib','xarray','rioxarray','pandas',
'scipy','astropy','scikit-learn','gdal-wheel>=3.4',
'geopandas','snaphu'],
    extras_require = {
        # optional speedups
        'optional': ['dask', 'pyinterp']
    }
)
