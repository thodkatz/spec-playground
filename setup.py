from setuptools import setup, find_packages

setup(
    name="myspec",
    version="1.0",
    packages=find_packages(),
    install_requires=["bioimageio.core==0.6.8", "bioimageio.spec==0.5.3.2", "torch"],
)
