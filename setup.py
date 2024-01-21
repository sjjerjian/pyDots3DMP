from setuptools import setup, find_packages
from pathlib import Path

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

long_description = ("README.md").read_text()
    
setup(
    name='pydots3DMP',
    version='0.1.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=requirements,
)