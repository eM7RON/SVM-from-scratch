# Support Vector Machines From Scratch

Just some exploration of SVMs; how they work and what they do. I keep these tutorials to act as a reference for myself and to educate others.

![](https://github.com/eM7RON/SVM-from-scratch/blob/master/img1.svg)

Outline:

1. Familiarise ourselves with constrained optimisation and the Lagrangian dual formulation.
2. Code a linear large margin classifier for linearly separable data.
3. Modify this code, which uses a hard margin to a soft margin.
4. Finally, use the kernel trick to map the data to a high-dimensional feature space, while calculating the necessary inner-products in the original data space.
5. Adapt SVM for multiclass problems
6. Discuss

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

python 3  
numpy  
scipy  
scikit-learn  
matplotlib  
scs  
cvxpy  

The following Python environment is recommended:

##### 3.6.7 (default, Dec  6 2019, 07:03:06) [MSC v.1900 64 bit (AMD64)]

<pre>
Name                      Version                   Build    Channel   
backcall                  0.1.0                      py_0    conda-forge  
blas                      1.0                         mkl   
ca-certificates           2019.11.28           hecc5488_0    conda-forge  
certifi                   2019.11.28               py36_0    conda-forge  
colorama                  0.4.3                      py_0    conda-forge  
cvxopt                    1.2.0            py36hdc3235a_0   
cvxpy                     1.0.25           py36h6538335_2    conda-forge  
cvxpy-base                1.0.25           py36h6538335_2    conda-forge  
cycler                    0.10.0                     py_2    conda-forge  
decorator                 4.4.1                      py_0    conda-forge  
dill                      0.3.1.1                  py36_0    conda-forge  
ecos                      2.0.7            py36h8c2d366_0   
entrypoints               0.3                   py36_1000    conda-forge  
fastcache                 1.1.0            py36hfa6e2cd_0    conda-forge  
freetype                  2.10.0               h563cfd7_1    conda-forge  
future                    0.18.2                   py36_0    conda-forge  
glpk                      4.65              h2fa13f4_1002    conda-forge  
gsl                       2.4               h631dd0c_1006    conda-forge  
icc_rt                    2019.0.0             h0cc432a_1   
icu                       64.2                 he025d50_1    conda-forge  
intel-openmp              2019.4                      245   
ipykernel                 5.1.4            py36h5ca1d4c_0    conda-forge  
ipython                   7.11.1           py36h5ca1d4c_0    conda-forge  
ipython_genutils          0.2.0                      py_1    conda-forge  
jedi                      0.16.0                   py36_0    conda-forge  
joblib                    0.14.1                     py_0    conda-forge  
jpeg                      9c                hfa6e2cd_1001    conda-forge  
jupyter_client            5.3.4                    py36_1    conda-forge  
jupyter_core              4.6.1                    py36_0    conda-forge  
kiwisolver                1.1.0            py36he980bc4_0    conda-forge  
libblas                   3.8.0                    14_mkl    conda-forge  
libcblas                  3.8.0                    14_mkl    conda-forge  
libclang                  9.0.1           default_hf44288c_0    conda-forge  
liblapack                 3.8.0                    14_mkl    conda-forge  
libpng                    1.6.37               h7602738_0    conda-forge  
libsodium                 1.0.17               h2fa13f4_0    conda-forge  
matplotlib                3.1.2                    py36_1    conda-forge  
matplotlib-base           3.1.2            py36h2981e6d_1    conda-forge  
mkl                       2019.4                      245   
mkl-service               2.3.0            py36hfa6e2cd_0    conda-forge  
multiprocess              0.70.9           py36hfa6e2cd_0    conda-forge  
numpy                     1.17.5           py36hc71023c_0    conda-forge  
openssl                   1.1.1d               hfa6e2cd_0    conda-forge  
osqp                      0.6.1            py36he350917_1    conda-forge  
parso                     0.6.0                      py_0    conda-forge  
pickleshare               0.7.5                 py36_1000    conda-forge  
pip                       20.0.2                   py36_1    conda-forge  
prompt_toolkit            3.0.3                      py_0    conda-forge  
pygments                  2.5.2                      py_0    conda-forge  
pyparsing                 2.4.6                      py_0    conda-forge  
pyqt                      5.12.3           py36h6538335_1    conda-forge  
pyqt5-sip                 4.19.18                  pypi_0    pypi  
pyqtwebengine             5.12.1                   pypi_0    pypi  
pyreadline                2.1                   py36_1001    conda-forge  
python                    3.6.7             he025d50_1006    conda-forge  
python-dateutil           2.8.1                      py_0    conda-forge  
pywin32                   225              py36hfa6e2cd_0    conda-forge  
pyzmq                     18.1.1           py36h16f9016_0    conda-forge  
qt                        5.12.5               h7ef1ec2_0    conda-forge  
scikit-learn              0.22.1           py36h7208079_1    conda-forge  
scipy                     1.3.1            py36h29ff71c_0    conda-forge  
scs                       2.1.1.2          py36h5b07068_3    conda-forge  
setuptools                45.1.0                   py36_0    conda-forge  
six                       1.14.0                   py36_0    conda-forge  
sqlite                    3.30.1               hfa6e2cd_0    conda-forge  
tornado                   6.0.3            py36hfa6e2cd_0    conda-forge  
tqdm                      4.42.0                     py_0    conda-forge  
traitlets                 4.3.3                    py36_0    conda-forge  
vc                        14.1                 h0510ff6_4   
vs2015_runtime            14.16.27012          hf0eaf9b_1   
wcwidth                   0.1.8                      py_0    conda-forge  
wheel                     0.34.2                   py36_0    conda-forge  
wincertstore              0.2                   py36_1003    conda-forge  
zeromq                    4.3.2                h6538335_2    conda-forge  
zlib                      1.2.11            h2fa13f4_1006    conda-forge  
</pre>

### Install instructions:

#### Create new conda environment from .yml file

Download and extract the repository. Navigate to the resulting folder in Anaconda prompt and use the following command:

###### conda env create -f conda_setup.yml

this creates the environment: C:\Anaconda3\envs\ML

The name and location of the environment may be altered by editing the conda_setup.yml file.

Alternatively, one can update an existing environment with :

###### conda env update --name myenv --file conda_setup.yml

where *myenv* is the name of the environment

#### Create a new environment using anaconda prompt

I encountered some issues trying to create a stable environment for the notebook. The main problems encountered were due to the scs convex optimization package not linking/detecting mkl/blas/lapack and the grid search sklearn function not parallelizing correctly. Although, it is quite possible that this problem is exclusive to users of Windows 10 64-bit OS like myself.

In the end I came up with this simple install. Simply run the following in anaconda prompt to create a new environment named ML (mac/linux users should write 'source' before 'conda' I think): 

###### conda create -n ML -c conda-forge -c defaults python=3.6.7 ipykernel numpy scipy scikit-learn matplotlib tqdm cvxopt cvxpy

'-c conda-forge -c defaults' specifies the priorities, descending from left to right, of the channels from which each package should be downloaded.

The new environment must be activated with:

###### conda activate ML

To make the new environment avaialable in Jupyter lab/notebooks:

###### python -m ipykernel install --user --name ML --display-name ML

## Authors

* **eM7RON (Simon Tucker)** - *Initial work* - [Github](https://github.com/eM7RON), [linkedin](https://www.linkedin.com/in/simon-tucker-21838372/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* **neal-o-r** - *Platt Scaling* - [Github](https://github.com/neal-o-r), [Platt Scaling](https://github.com/neal-o-r/platt)
