from setuptools import find_packages, setup

setup(
    name="argus",
    version="0.1.0",
    description="ARGUS — AI-powered smart glasses runtime for the visually impaired",
    packages=find_packages(include=["argus", "argus.*"]),
    python_requires=">=3.10",
    entry_points={"console_scripts": ["argus=argus.__main__:main"]},
)
