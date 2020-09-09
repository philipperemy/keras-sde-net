from setuptools import setup, find_packages

VERSION = '1.0'

setup(
    name='sdenet',
    version=VERSION,
    author='Philippe Remy',
    license='MIT',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        'numpy==1.19.1',
        'tensorflow==2.3.0',
        'tensorflow_addons==0.11.2'
    ],
    python_requires='>=3',
)
