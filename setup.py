import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deep-expressions",
    version="0.0.1",
    author="Humberto da Silva Neto",
    author_email="humberto.nt4@gmail.com",
    description="A Deep Learning toolkit for Facial Expressions Recognition (FER)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deepexpressions/DeepExpressions",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'opencv-python>=3.4.0', 
        'Pillow>=6.0.0', 
    ]
)