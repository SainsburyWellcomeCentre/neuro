from setuptools import setup, find_namespace_packages

requirements = [
    "numpy",
    "scikit-image>=0.14.0,<0.17.0",
    "pandas>=0.25.1,<=0.25.3",
    "napari[pyqt5]",
    "brainrender==0.4.0.0",
    "imlib >= 0.0.26",
    "brainio >= 0.0.19",
    "dask >= 2.15.0",
]


setup(
    name="neuro",
    version="0.0.19",
    description="Visualisation and analysis of brain imaging data",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black",
            "pytest-cov",
            "pytest",
            "gitpython",
            "coverage",
            "bump2version",
            "pre-commit",
            "flake8",
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
            "cellfinder_view = neuro.visualise.viewer:main",
            "fibre_track = "
            "neuro.segmentation.lesion_and_track_tools.fiber_tract_viewer:main",
            "manual_seg = "
            "neuro.segmentation.manual_segmentation.segment:main",
        ]
    },
    zip_safe=False,
)
