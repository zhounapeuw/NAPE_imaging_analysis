Anaconda and Jupyter Notebook Installation Guide (For Calcium Imaging Analysis)
===============================================================================

This tutorial will walk you through how to install python, Anaconda Prompt/Navigator, the necessary prerequisites to run the calcium imaging preprocessing analysis. The steps involve:

.. highlight:: rst

.. role:: python(code)
    :language: python

* Install Anaconda: this is a graphical and command line interface that allows you to manage multiple, unique instances (called environments) of python that can be tailored to different projects. Think of Anaconda as a drawer file organizer where each folder (ie. environment) pertains to a specific project/topic.
* Set up an Anaconda environment for the specific calcium imaging preprocessing project: An environment is a directory/instance that contains a collection of python packages that the user can customize and tailor to a specific project. This is especially important if you have multiple projects that require different versions of python or conflicting packages/libraries that must be separated.
* Open and run jupyter notebook, application that allows for editing, running, and prototyping python code.

Unfortunately and fortunately (it’s beneficial to learn some coding!) for you, this will involve some tinkering with the Anaconda command prompt. To facilitate this process, all commands to be executed in the command prompt are bolded and any text that needs to be changed by you is bolded and italicized.

1. Download the Anaconda Installer: https://www.anaconda.com/distribution/#windows and run the installer.
   a. The 64-bit graphical installer is recommended for most PCs
   b.	Choose Python 3.7 version for most up-to-date python version. Note: You can still install a Python 2.7 environment in Anaconda.

.. image:: _images/anaconda_sima_install/1_anaconda_website.png

2.	Follow the installer prompts: Hit “Next”, “I Agree”, “Next” for Just Me (Installation Type)

.. image:: _images/anaconda_sima_install/2_anaconda_setup_1.png

3.	Indicate the path to install Anaconda and hit “Next”

.. image:: _images/anaconda_sima_install/3_anaconda_setup_2.png

4)	On the Advanced Installation Options window, keep the default settings and hit “Install” and continue through the installation until the end (ie. Hit “Next”s)

Starting Anaconda and installing an environment:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5)	Search for “Anaconda” in the start menu and click “Anaconda Prompt

.. image:: _images/anaconda_sima_install/4_open_prompt.png

This is how the anaconda prompt looks like at the start. The current environment is in parentheses and the following text indicates what directory you currently are in (equivalent to if you had the finder/explorer window open to that specific folder).
* Note: an environment is a directory that contains a collection of python packages that the user can customize and tailor to a specific project. One can create, edit, and delete any number of environments as he or she chooses.

.. image:: _images/anaconda_sima_install/5_prompt.png

6) Download the sima_env.yml file in this link: https://github.com/zhounapeuw/NAPE_imaging_analysis/blob/master/sima_env.yml
*  To do this, you need to right click the “raw” button and click “Save target as” and save the file to an easy-to-access folder (throughout this walkthrough, it will be in the Downloads folder).

.. image:: _images/anaconda_sima_install/6_get_env.png

7) Take note of where the sima_env.yml file is located (in this example it is in my downloads folder, but it can be anywhere you want as long as you provide the correct path to the file during installation).
*  Take note of the full path to where the sima_env.yml is in. If it is in the downloads folder, the path will be similar to “C:/Users/user_name/Downloads” where user_name may differ across computers based on what the user name is. Please replace user_name with your PC’s username.
*  Note: the sima_env.yml file contains installation information to recreate the same environment and installed packages to run the analyses.
*  Note: the sima_env.yml file can be deleted once everything has been installed and the analysis scripts work
*  Note: the python version of this environment is 2.7, which is required for SIMA (the motion correction package) to run.

4. Replace :python:`{PATH_TO_THE_FILE}` with path of :python:`environment.yml` and run :python:`conda env create -f {PATH_TO_THE_FILE}\environment.yml`. In this case, :python:`{PATH_TO_THE_FILE}` is :python:`D:\NAPE_2pBenchmark`

