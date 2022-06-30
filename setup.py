import os
import re

from setuptools import find_packages
from setuptools import setup


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


def get_install_requires():
    return ["art", "imgviz", "oct_unet==0.7.0", "pillow==9.0.1", "prettytable", "typeguard"]


def get_long_description():
    return "This should contain a long description"


def get_version():
    filename = "oct_segmenter/__init__.py"
    with open(filename) as f:
        match = re.search(
            r"""^__version__ = ['"]([^'"]*)['"]""", f.read(), re.M
        )
    if not match:
        raise RuntimeError("{} doesn't contain __version__".format(filename))
    version = match.groups()[0]
    return version


def main():
    version = get_version()

    setup(
        name="oct-segmenter",
        version=version,
        packages=find_packages(exclude=["unet*"]),
        description="Image Segmentation Tool for Mice OCT scans",
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        author="BioTeam, Inc.",
        author_email="bruno@bioteam.net",
        url="https://www.bioteam.net",
        install_requires=get_install_requires(),
        license="GPLv3",
        keywords="Image Segmentation, Machine Learning",
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Researchers",
            "Natural Language :: English",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3.9",
        ],
        package_data={
            "oct_segmenter": package_files("oct_segmenter/data/models/")
        },
        entry_points={
            "console_scripts": [
                "oct-segmenter=oct_segmenter.__main__:main",
            ],
        },
        data_files=[],
    )

if __name__ == "__main__":
    main()