# UW NAPE Imaging Analysis/Preprocessing

Welcome to the UW NAPE (Neurobiology of Addiction, Pain, and Emotion) Center's calcium imaging analysis repository. You will find code for preprocessing calcium imaging (primarily 2-photon, GRIN lens-based deep brain imaging) data. 

<img width="300" alt="logo" src="https://github.com/zhounapeuw/NAPE_imaging_analysis/blob/master/docs/_images/logo.jpg">
Image created by Morgan Alexander

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

Python 2.7  
h5py==2.10.0  
future==0.14.3  
numpy==1.16.5  
Pillow==6.2.0  
scikit-image==0.14.5  
scikit-learn==0.20.4  
scipy==1.2.2  
Shapely==1.6.4.post2  

### Installation and Usage

Follow the links below for step by step tutorials on how to install anaconda (python), the environment with the prerequisites, and run the code:

* [Docs Install Guide](https://zhounapeuw.github.io/NAPE_imaging_analysis/install_anaconda_sima.html#)
* [Ca2+ Imaging Preprocessing Pipeline Video Tutorial](https://www.youtube.com/watch?v=j-fUlq6L92U&t=699s)

The main jupyter notebook script for the preprocessing pipeline **(main_parallel.ipynb) resides in the napeca folder**.
The main Bruker/tiff prepreprocessing script is in the napeca/prepreprocess folder.

## Built With

* [SIMA](https://github.com/losonczylab/sima) - Adapted motion correction algorithm; Kaifosh et al., 2014

## Authors

* **Zhe Charles Zhou** - *Initial work* - [Zhou NAPE UW](https://github.com/zhounapeuw)
* **Tanish Kapur** - *Initial sphinx setup* - [Tanish Kapur](https://github.com/tan33sh)
* **Alex De Lecea** - *Documentation Development* - [Alex De Lecea](https://github.com/Alex-de-Lecea)

## 

If the code and resources outlined here are helpful to your project, we encourage you to cite/acknowledge the UW NAPE Imaging Core. Thank you!

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

