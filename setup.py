import os
from setuptools import find_packages, setup

# Load version number
__version__ = ''

src_dir = os.path.abspath(os.path.dirname(__file__))
version_file = os.path.join(src_dir, 'pp3', '_version.py')

with open(version_file) as fd:
    exec(fd.read())

# Load README
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pp3',
    version=__version__,
    author='Kyle Swanson, Jeremy Wohlwend, Mert Yuksekgonul',
    author_email='swansonk.14@gmail.com, jeremy.wohlwend@gmail.com, merty@stanford.edu',
    description='3D Protein Probing',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/swansonk14/3d_protein_probing',
    download_url=f'https://github.com/swansonk14/3d_protein_probing/archive/refs/tags/v_{__version__}.tar.gz',
    license='MIT',
    packages=find_packages(),
    package_data={'pp3': ['py.typed']},
    install_requires=[],  # TODO: add dependencies
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        "Typing :: Typed"
    ],
    keywords=[
        '3d protein probing'
    ]
)
