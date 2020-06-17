import setuptools
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        donwload_and_extract()

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        donwload_and_extract()


def donwload_and_extract():
    import gdown
    import zipfile

    url = 'https://drive.google.com/uc?id=1QH_-bazUQNbsyK7WA-4XqBJJpqXL1W36'
    output = 'data/bovespa.zip'
    gdown.download(url, output, quiet=False)
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('data/')

def requirements_list():
    list_of_req = []
    with open('requirements.txt') as req:
        for line in req:
            list_of_req.append(line)

    return list_of_req

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="b3data",
    version="0.0.1",
    author="EmpyreanAI",
    author_email="author@example.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EmpyreanAI/B3Data",
    packages=setuptools.find_packages(),
    install_requires=requirements_list(),
    include_package_data=True,
    package_data={"":['data/*.csv']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
    python_requires='>=3.6',
)