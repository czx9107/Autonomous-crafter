import pathlib

import setuptools


setuptools.setup(
    name='aidr-crafter',
    version='0.1.0',
    description='Standalone Crafter-style environment extracted from Conan for RL experiments.',
    long_description=pathlib.Path('README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    package_data={'crafter': ['data.yaml', 'assets/*.png']},
    include_package_data=True,
    install_requires=[
        'numpy',
        'imageio',
        'pillow',
        'opensimplex',
        'ruamel.yaml',
        'gym',
    ],
    extras_require={
        'gui': ['pygame'],
        'training': ['stable_baselines3', 'sb3-contrib', 'torch', 'gymnasium', 'tensorboard'],
    },
    entry_points={
        'console_scripts': [
            'aidr-crafter-random=crafter.run_random:main',
            'aidr-crafter-gui=crafter.run_gui:main',
        ],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
