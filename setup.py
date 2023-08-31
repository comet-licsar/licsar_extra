from setuptools import setup

setup(
    name='licsar_extra',
    version='0.1',
    description='LiCSAR Extra set of tools',
    url='https://github.com/comet-licsar/licsar_extra',
    author='Milan Lazecky',
    author_email='M.Lazecky@leeds.ac.uk',
    license='GNU-GPL3.0',
    packages=['licsar_extra'],
    package_dir = {'licsar_extra':'python'},
    install_requires = ['numpy<2.0',
'os','glob','shutil','subprocess','re','time',
'matplotlib','xarray','rioxarray','pandas',
'scipy','astropy','scikit-learn','gdal>=2.4']
)
