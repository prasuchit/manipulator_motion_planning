from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="manipulator_motion_planning",
    version="0.1.0",
    description="Motion planning algorithms for robotic manipulators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Prasanth Sengadu Suresh",
    url="https://github.com/prasuchit/manipulator_motion_planning",

    # only look inside src for the main package
    packages=find_packages(where="src"),

    # tells setuptools that package root lives in src/
    package_dir={"": "src"},

    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    extras_require={
        "dev": ["black", "isort", "ruff", "pytest"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
)