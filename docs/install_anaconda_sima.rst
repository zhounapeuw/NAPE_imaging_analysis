Anaconda and Jupyter Notebook Installation Guide (For Calcium Imaging Analysis)
====================================

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

.. image:: 1_anaconda_website.png

4. Replace :python:`{PATH_TO_THE_FILE}` with path of :python:`environment.yml` and run :python:`conda env create -f {PATH_TO_THE_FILE}\environment.yml`. In this case, :python:`{PATH_TO_THE_FILE}` is :python:`D:\NAPE_2pBenchmark`

