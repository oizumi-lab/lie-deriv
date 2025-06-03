# setup.py （lie-deriv プロジェクトルートに置く）

from setuptools import setup, find_packages

setup(
    name='lie-deriv',
    version='0.1.0',
    description='Lie‐equivariance metrics and related utilities (including StyleGAN3 integration)',
    author='Chanseok Lim',
    author_email='ch-lim@g.ecc.u-tokyo.ac.jp',
    url='https://github.com/oizumi-lab/lie-deriv.git',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'tqdm',
    ],
    python_requires='>=3.10',
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)

