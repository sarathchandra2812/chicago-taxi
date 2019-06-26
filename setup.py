from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = []

setup(
    name='taxi_fare_forecaster',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='This model forecasts taxi fares.'
)