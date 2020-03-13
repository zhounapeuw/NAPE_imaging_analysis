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

1) Download the Anaconda Installer: https://www.anaconda.com/distribution/#windows and run the installer.
   a. The 64-bit graphical installer is recommended for most PCs
   b. Choose Python 3.7 version for most up-to-date python version. Note: You can still install a Python 2.7 environment in Anaconda.

.. image:: _images/anaconda_sima_install/1_anaconda_website.png

2)	Follow the installer prompts: Hit “Next”, “I Agree”, “Next” for Just Me (Installation Type)

.. image:: _images/anaconda_sima_install/2_anaconda_setup_1.png

3)	Indicate the path to install Anaconda and hit “Next”

.. image:: _images/anaconda_sima_install/3_anaconda_setup_2.png

4)	On the Advanced Installation Options window, keep the default settings and hit “Install” and continue through the installation until the end (ie. Hit “Next”s)

Starting Anaconda and installing an environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

5)	Search for “Anaconda” in the start menu and click “Anaconda Prompt

.. image:: _images/anaconda_sima_install/4_open_prompt.png

This is how the anaconda prompt looks like at the start. The current environment is in parentheses and the following text indicates what directory you currently are in (equivalent to if you had the finder/explorer window open to that specific folder).
* Note: an environment is a directory that contains a collection of python packages that the user can customize and tailor to a specific project. One can create, edit, and delete any number of environments as he or she chooses.

.. image:: _images/anaconda_sima_install/5_prompt.png

6) Download the sima_env.yml file in this link: https://github.com/zhounapeuw/NAPE_imaging_analysis/blob/master/sima_env.yml
*  To do this, you need to right click the “raw” button and click “Save target as” and save the file to an easy-to-access folder (throughout this walkthrough, it will be in the Downloads folder).

.. image:: _images/anaconda_sima_install/6_get_env.png

7) Take note of where the sima_env.yml file is located (in this example it is in my downloads folder, but it can be anywhere you want as long as you provide the correct path to the file during installation).
*  Take note of the full path to where the sima_env.yml is in. If it is in the downloads folder, the path will be similar to **“C:/Users/user_name/Downloads”** where **user_name** may differ across computers based on what the user name is. Please replace *user_name* with your PC’s username.
*  Note: the sima_env.yml file contains installation information to recreate the same environment and installed packages to run the analyses.
*  Note: the sima_env.yml file can be deleted once everything has been installed and the analysis scripts work
*  Note: the python version of this environment is 2.7, which is required for SIMA (the motion correction package) to run.

.. image:: _images/anaconda_sima_install/7_env_path.png

8) Copy, paste, and execute the following code into the anaconda prompt to recreate a new environment from the sima_env.yml file: **conda env create -n sima_env -f C:/Users/user_name/Downloads/sima_env.yml** (remember to replace the user_name)
* Note, you can also navigate to the Downloads folder yourself using the **cd** command and simply execute **conda env create -n sima_env -f sima_env.yml**

.. image:: _images/anaconda_sima_install/8_create_env.png

Once the environment installer runs through, you should see a list of all the conda and python packages successfully installed.

.. image:: _images/anaconda_sima_install/9_env_installed.png

9) Thus far, we have been operating under the default, base environment; we need to switch over to the new sima_env environment we just created. We do this by typing and executing: **conda activate sima_env**.

::

    conda activate sima_env

.. image:: _images/anaconda_sima_install/10_activate_env.png

10) If you encounter an error that contains: LookupError: unknown encoding: cp65001 , you will need to execute the following line: ``set PYTHONIOENCODING=UTF-8``

11) Navigate to the following link https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely and download "Shapely-1.6.4.post2-cp27-cp27m-win_amd64.whl" .

Then in anaconda prompt, run ``pip install C:/Users/user_name/Downloads/Shapely-1.6.4.post2-cp27-cp27m-win_amd64.whl`` (remember to replace the user_name) .

.. image:: _images/anaconda_sima_install/11_install_shapely.png

12) To complete the environment installation, execute ``pip install sima``

.. image:: _images/anaconda_sima_install/12_install_sima.png

Using jupyter notebook to edit and run code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

13) Download the SIMA_motion_correction.ipynb script from this link: https://github.com/zhounapeuw/NAPE_imaging_analysis/blob/master/SIMA_motion_correction.ipynb

.. image:: _images/anaconda_sima_install/13_download_mc_code.png

14) Take note of where the analysis jupyter notebook or python script resides (We have it in the Downloads folder for this demo)

.. image:: _images/anaconda_sima_install/14_mc_code_dir.png

Then navigate to that folder in the anaconda prompt : ``cd C:/Users/user_name/Downloads/`` (remember to replace the user_name)

.. image:: _images/anaconda_sima_install/15_cd_to_code.png

15) Execute ``jupyter notebook`` and an instance of jupyter notebook will start up in your web browser.
* Jupyter notebook is a powerful application that allows for editing and running python code. Anaconda boots up an instance of python that can be interacted with via the jupyter notebook web client.
* The first page that opens in your browser will show the files in your current directory specified in the Anaconda prompt. Files with the ipynb (iPython notebook) extension can be clicked and will open the notebook.

.. image:: _images/anaconda_sima_install/16_jupyter_open.png

Then the following window will open in your default browser:

.. image:: _images/anaconda_sima_install/17_jupyter_notebook.png

A jupyter notebook consists of cells where one can write and execute code. Typically the first cell contains lines for importing packages and dependencies. For example, for us to use the SIMA library and its functions, we must have an import sima line.
* To run a cell, the easiest way is to press shift + enter
* Refer to this guide for more details on how to use jupyter notebook: https://www.codecademy.com/articles/how-to-use-jupyter-notebooks

.. image:: _images/anaconda_sima_install/18_mc_code.png

Troubleshooting
~~~~~~~~~~~~~~~

A) If you encounter the following problem during environment installation:
LinkError: post-link script failed for package defaults::qt-5.6.2-vc9hc26998b_12
location of failed script: C:\Users\stuberadmin\Anaconda3\envs\tmp_sima\Scripts\.qt-post-link.bat
You will need to search “edit the system environment variables” in the search bar and add this path: C:\Windows\System32\ to the current user’s environmental path variables.

.. image:: _images/anaconda_sima_install/19_env_var.png

.. image:: _images/anaconda_sima_install/20_env_var_2.png



4. Replace :python:`{PATH_TO_THE_FILE}` with path of :python:`environment.yml` and run :python:`conda env create -f {PATH_TO_THE_FILE}\environment.yml`. In this case, :python:`{PATH_TO_THE_FILE}` is :python:`D:\NAPE_2pBenchmark`

