from setuptools import setup, find_namespace_packages

requirements = [
    "numpy",
    "scikit-image",
    "pandas<=0.25.3,>=0.25.1",
    "napari",
    "brainrender",
    "imlib >= 0.0.20",
    "brainio => 0.0.12",
]


setup(
    name="neuro",
    version="0.0.9",
    description="Visualisation and analysis of brain imaging data",
    install_requires=requirements,
    extras_require={
        "dev": [
            "sphinx",
            "recommonmark",
            "sphinx_rtd_theme",
            "pydoc-markdown",
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coveralls",
            "coverage<=4.5.4",
        ]
    },
    python_requires=">=3.6, <3.8",
    packages=find_namespace_packages(exclude=("docs", "tests*")),
    include_package_data=True,
    url="https://github.com/SainsburyWellcomeCentre/neuro",
    author="Adam Tyson",
    author_email="adam.tyson@ucl.ac.uk",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        "console_scripts": [
            "points_to_brainrender = "
            "neuro.points.points_to_brainrender:main",
            "heatmap = neuro.heatmap.heatmap:cli",
            "amap_vis = neuro.visualise.amap_vis:main",
            "fibre_track = "
            "neuro.segmentation.lesion_and_track_tools.fiber_tract_viewer:main",
            "manual_region_seg = "
            "neuro.segmentation.manual_region_segmentation.segment:main",
        ]
    },
    zip_safe=False,
)
