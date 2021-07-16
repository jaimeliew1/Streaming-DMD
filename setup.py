# https://python-packaging.readthedocs.io
from setuptools import setup

setup(
    name="sDMD",
    version="1.0",
    description="Streaming Dynamic Mode Decomposition",
    author="Jaime Liew",
    author_email="jyli@dtu.dk",
    # license="",
    packages=["sDMD"],
    install_requires=["pytest"],
    zip_safe=False,
    include_package_date=True,
)
