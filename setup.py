import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="junengine",
    version="0.1.0",
    author="David Steele",
    author_email="david.arthur.steele@gmail.com",
    description="JunEngine Online Data Science Marketplace",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/darthursteele/junengine",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT Licensess",
        "Operating System :: OS Independent",
    ],
)

