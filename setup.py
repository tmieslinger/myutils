from setuptools import setup, find_packages


setup(
    name='myutils',
    version='0.1dev',
    description='Utility functions to work with ASTER.',
    author='Theresa Mieslinger',
    author_email='theresa.mieslinger@gmail.com',
    license='GPLv3',
    packages=find_packages(),
    install_requires=[
        'netCDF4',
        'numpy',
        'typhon',
	'xarray',
	'cdsapi',
    ]
)
