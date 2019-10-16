from setuptools import setup, find_packages
setup(
    name="multilum",
    version="0.1",
    packages=find_packages(),

    test_suite="tests.test_multilum",

    author="Lukas Murmann",
    author_email="lmurmann@mit.edu",
    description="Multi Illumination Image Sets SDK",
    keywords="graphics vision multi-illumination",
    url="https://projects.csail.mit.edu/multi-illumination",
    project_urls={
        "Source Code": "https://github.com/lmurmann/multilum_sdk",
    },
)
