from setuptools import setup, find_namespace_packages

requirements = [
    "numpy",
    "scikit-image",
    "pandas",
    "napari",
    "brainrender",
    "imlib",
    "brainio",
]


setup(
    name="neuro",
    version="0.0.5",
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
        ]
    },
    zip_safe=False,
)
