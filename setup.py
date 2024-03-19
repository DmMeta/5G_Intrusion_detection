from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as req:
       return [line.strip() for line in req]

setup(
    name="AnalysisTraining",
    version = "0.1",
    packages= find_packages(where = "package_src"),
    package_dir = {"": "package_src"},
    install_requires = read_requirements()
)