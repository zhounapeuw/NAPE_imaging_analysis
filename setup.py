import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="napeca-zhounapeuw", # Replace with your own username
    version="0.0.10",
    author="Zhe Charles Zhou",
    author_email="zhouzc@uw.edu",
    description="NAPE calcium imaging preprocessing pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhounapeuw/NAPE_imaging_analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    install_requires=['PyWavelets==1.0.3',
                      'Shapely==1.4.1',
                      'h5py==2.10.0',
                      'future==0.14.3',
                      'numpy==1.16.5',
                      'Pillow==9.0.1',
                      'scikit-image==0.14.5',
                      'scikit-learn==0.20.4',
                      'scipy==1.2.3',
                      'sima==1.3.2'

                      ],  # Optional

)