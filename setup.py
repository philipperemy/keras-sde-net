from setuptools import setup, find_packages

from sdenet import VERSION

setup(
    name='SDE Net - Keras',
    version=VERSION,
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=open('requirements.txt').read().strip().split('\n'),
    python_requires='>=3',
)
