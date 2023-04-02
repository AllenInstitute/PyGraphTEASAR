from setuptools import setup, find_packages
import re
import os
import codecs

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "r") as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    version=find_version("GraphTEASAR", "__init__.py"),
    name="GraphTEASAR",
    description="A python library for running the TEASAR algorithm on spatial graphs for skeletonization.",
    author="Forrest Collman",
    author_email="forrestc@alleninstute.org,caseys@alleninstitute.org,svenmd@princeton.edu,",
    url="https://github.com/AllenInstitute/PyGraphTEASAR",
    packages=find_packages(where="."),
    include_package_data=True,
    install_requires=required,
    setup_requires=["pytest-runner"],
)
