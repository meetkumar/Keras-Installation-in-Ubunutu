# Keras-using-Tensorflow-gpu-backend-Installation-in-Ubunutu
Basic installation of Keras, along with NVIDIA drivers for GPU, CUDA toolkit and CuDNN.

# Start by installing pip

$ sudo apt-get update && sudo apt-get -y upgrade

$ sudo apt-get install python-pip #Python 2.7

$ sudo apt-get install python3-pip #Python 3.6

Verify pip

$ pip -V

pip 9.0.1 from /usr/lib/python2.7/dist-packages (python 2.7)


# 1: Now for Keras backend Tensorflow to support GPU you need to do following steps:
Install Virtualenv for python and python 3:

$ sudo apt-get update

$ sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python-virtualenv swig python-wheel libcurl3-dev

Create a Virtualenv environment in the directory for python and python 3:

#for python 2

$ virtualenv --system-site-packages -p python ~/tensorflow

#for python 3 

$ virtualenv --system-site-packages -p python3 ~/tensorflow

(Note: To delete a virtual environment, just delete its folder.  For example, In our cases, it would be rm -rf keras-tf-venv or rm -rf keras-tf-venv3.)

(Note: Don't install tensorflow-gpu using pip it will not work properly and also don't do pip install tensorflow until mentioned (To uninstall tensorflow, just type pip uninstall tensorflow tensorflow-gpu))

# 2: Update & Install NVIDIA Drivers (skip this if you do not need to TensorFlow GPU version)
2a. Determine the latest version of Nvidia driver available for your graphics card

a. Visit the graphics drivers PPA homepage (https://launchpad.net/~graphics-drivers/+archive/ubuntu/ppa) and determine the latest versions of Nvidia drivers available which is ‘nvidia-387’ as of January 17, 2018.

b. Verify that your graphics card is capable of running the latest drivers. You can search on this link (http://www.nvidia.com/object/unix.html) to determine if your graphics card is supported by a driver version. Don’t be so particular about the version part after the dot (after nvidia-387.xxx), just make sure you’re supported on the main version 387.

2b. Remove older Nvidia driver

If your graphic is supported, you can go ahead and remove all previously installed Nvidia drivers on your system. Enter the following command in terminal.

$ sudo apt-get purge nvidia*

2c. Add the graphics drivers PPA

Let us go ahead and add the graphics-driver PPA -

$ sudo add-apt-repository ppa:graphics-drivers

And update

$ sudo apt-get update

2d. Install (and activate) the latest Nvidia graphics drivers.Enter the following command to install the version of Nvidia graphics supported by your graphics card -

$ sudo apt-get install nvidia-387(or which ever version)

Verify Driver correctly installed

$ nvidia-smi

(Note : If nothing shows up don't worry, just reboot, in the menu choose "change secure boot options", put the password you previously chose and disable the secure boot.)
For more details : http://www.linuxandubuntu.com/home/how-to-install-latest-nvidia-drivers-in-linux

# 3 : Install NVIDIA CUDA Toolkit 8.0 (Skip if not installing with GPU support)

(Note: If you have older version of CUDA and cuDNN installed, check the post for uninstallation.  How to uninstall CUDA Toolkit and cuDNN under Linux? (02/16/2017) (http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#runfile-uninstallation))

Remember to download CUDA Toolkit 8.0 as tensorflow-gpu doesn't support CUDA Toolkit 9.1
(Note: You can download CUDA Toolkit 8.0 by scrolling down on the official nvidia cuda toolkit download page (https://developer.nvidia.com/cuda-downloads) and selecting Legacy Releases (at the bottom of the page) > CUDA Toolkit 8.0 GA2 (Feb 2017))

$ cd ~/Downloads # or directory to where you downloaded file

$ sudo sh cuda_8.0.44_linux.run  # hold s to skip

This will install cuda into: /usr/local/cuda-8.0

MAKE SURE YOU SAY NO TO INSTALLING NVIDIA DRIVERS! (Very important, If you answer yes, the GTX 1080 387 driver will be overwritten.) 

Also make sure you select yes to creating a symbolic link to your cuda directory.

(FYI, the following is the questions to be asked.)

The following contains specific license terms and conditions
for four separate NVIDIA products. By accepting this
agreement, you agree to comply with all the terms and
conditions applicable to the specific product(s) included
herein.

Do you accept the previously read EULA?
accept/decline/quit: accept

Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 361.62?
(y)es/(n)o/(q)uit: n

Install the CUDA 8.0 Toolkit?
(y)es/(n)o/(q)uit: y

Enter Toolkit Location
[ default is /usr/local/cuda-8.0 ]:

Do you want to install a symbolic link at /usr/local/cuda?
(y)es/(n)o/(q)uit: y

Install the CUDA 8.0 Samples?
(y)es/(n)o/(q)uit: y

Enter CUDA Samples Location
[ default is /home/liping ]:

Installing the CUDA Toolkit in /usr/local/cuda-8.0 …
Installing the CUDA Samples in /home/liping …
Copying samples to /home/liping/NVIDIA_CUDA-8.0_Samples now…
Finished copying samples.

(Note: Don't forget to update your libraries into PATH mentioned in post installations in the link)

For more information : https://developer.download.nvidia.com/compute/cuda/9.1/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf

# 4 : Install NVIDIA cuDNN
Once the CUDA Toolkit is installed, download cuDNN v6.0 for Cuda 8.0 from NVIDIA website (Note that you will be asked to register an NVIDIA developer account in order to download) and extract into /usr/local/cuda via:

4a.  Install Runtime library

$ sudo dpkg -i $(runtime library deb)
 
4b.  Install developer library

$ sudo dpkg -i $(developer library deb)
 
4c.  Install code samples and user guide

$ sudo dpkg -i $(document library deb)

(Note: Download cuDNN v6.0 not cuDNN v7.0 as tensorflow-gpu doesn't support it)

For more information : https://developer.nvidia.com/cudnn

# 5: Install TensorFlow

Before installing TensorFlow and Keras, be sure to activate your python virtual environment first.

#for python 2

$ source ~/keras-tf-venv/bin/activate  # If using bash
(keras-tf-venv)$  # Your prompt should change

#for python 3

$ source ~/keras-tf-venv3/bin/activate  # If using bash

(keras-tf-venv3)$  # Your prompt should change

Install TensorFlow using one of the following commands (https://www.tensorflow.org/install/install_linux#InstallingVirtualenv):

(keras-tf-venv)$ pip install --upgrade tensorflow   # Python 2.7; CPU support (no GPU support)

(keras-tf-venv3)$ pip3 install --upgrade tensorflow   # Python 3.n; CPU support (no GPU support)

(keras-tf-venv)$ pip install --upgrade tensorflow-gpu  # Python 2.7;  GPU support

(keras-tf-venv3)$ pip3 install --upgrade tensorflow-gpu # Python 3.n; GPU support

Note: If the commands for installing TensorFlow given above failed (typically because you invoked a pip version lower than 8.1), install TensorFlow in the active virtualenv environment by issuing a command of the following format:

(keras-tf-venv)$ pip install --upgrade TF_PYTHON_URL   # Python 2.7

(keras-tf-venv3)$ pip3 install --upgrade TF_PYTHON_URL  # Python 3.N

where TF_PYTHON_URL identifies the URL of the TensorFlow Python package. The appropriate value of TF_PYTHON_URLdepends on the operating system, Python version, and GPU support. Find the appropriate value for TF_PYTHON_URL for your system here. For example, if you are installing TensorFlow for Linux, Python 2.7, and CPU-only support, issue the following command to install TensorFlow in the active virtualenv environment: (see below for examples. Note that check here to get the latest version for your system.)

#for python 2.7 -- CPU only

(keras-tf-venv)$pip install --upgrade \ https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl

#for python 2.7 -- GPU support

(keras-tf-venv)$pip install --upgrade \ https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp27-none-linux_x86_64.whl

#for python 3.5 -- CPU only

(keras-tf-venv3)$ pip3 install --upgrade \
https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp35-cp35m-linux_x86_64.whl

#for python 3.5 -- GPU support

(keras-tf-venv3)$ pip3 install --upgrade \ 
https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp35-cp35m-linux_x86_64.whl

Validate your TensorFlow installation. (as I just installed GPU tensorflow, so if you install CPU TensorFlow, the output might be slightly different.)

#For Python 2.7

(keras-tf-venv) :~$ python

Python 2.7.12 (default, Nov 19 2016, 06:48:10) 
[GCC 5.4.0 20160609] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2017-08-01 14:28:31.257054: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 14:28:31.257090: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 14:28:31.257103: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 14:28:31.257114: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 14:28:31.257128: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 14:28:32.253475: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:03:00.0
Total memory: 11.90GiB
Free memory: 11.75GiB
2017-08-01 14:28:32.253512: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0 
2017-08-01 14:28:32.253519: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0: Y 
2017-08-01 14:28:32.253533: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:03:00.0)

>>> print(sess.run(hello))
Hello, TensorFlow!
>>> exit()
(keras-tf-venv) :~$ 


#for python 3

(keras-tf-venv3) :~$ python3

Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> hello = tf.constant('Hello, TensorFlow!')
>>> sess = tf.Session()
2017-08-01 13:54:30.458376: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 13:54:30.458413: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 13:54:30.458425: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 13:54:30.458436: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 13:54:30.458448: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
2017-08-01 13:54:31.420661: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
name: TITAN X (Pascal)
major: 6 minor: 1 memoryClockRate (GHz) 1.531
pciBusID 0000:03:00.0
Total memory: 11.90GiB
Free memory: 11.75GiB
2017-08-01 13:54:31.420692: I tensorflow/core/common_runtime/gpu/gpu_device.cc:961] DMA: 0 
2017-08-01 13:54:31.420699: I tensorflow/core/common_runtime/gpu/gpu_device.cc:971] 0: Y 
2017-08-01 13:54:31.420712: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: TITAN X (Pascal), pci bus id: 0000:03:00.0)
>>> print(sess.run(hello))
b'Hello, TensorFlow!'
>>> exit() 
(keras-tf-venv3) :~$
If you see the output as below, it indicates your TensorFlow was installed correctly.

Hello, TensorFlow!


# 6: Install Keras

(Note: Be sure that you activated your python virtual environment before you install Keras.)

Installing Keras is even easier than installing TensorFlow.

First, let’s install a few dependencies:

#for python 2

$ pip install numpy scipy

$ pip install scikit-learn

$ pip install pillow

$ pip install h5py

#for python 3

$ pip3 install numpy scipy

$ pip3 install scikit-learn

$ pip3 install pillow

$ pip3 install h5py

Install keras:

$ pip install keras #for python 2

$ pip3 install keras # for python 3

Keras is now installed on your Ubuntu 16.04.

# 7: Verify that your keras.json file is configured correctly
Let’s now check the contents of our keras.json  configuration file. You can find this file at ~/.keras/keras.json .
use nano to open and edit the file.

$ nano ~/.keras/keras.json

The default values should be something like this:

{
 "epsilon": 1e-07,
 "backend": "tensorflow",
 "floatx": "float32",
 "image_data_format": "channels_last"
}

# Can’t find your keras.json file?

On most systems the keras.json  file (and associated subdirectories) will not be created until you open up a Python shell and directly import the keras  package itself.

If you find that the ~/.keras/keras.json  file does not exist on your system, simply open up a shell, (optionally) access your Python virtual environment (if you are using virtual environments), and then import Keras:

#for python 2

$ python

>>> import keras
>>> quit()

#for python 3

$ python3

>>> import keras
>>> quit()

From there, you should see that your keras.json  file now exists on your local disk.

If you see any errors when importing keras  go back to the top of step 4 and ensure your keras.json  configuration file has been properly updated.

Note: each time you would like to use Keras, you need to activate the virtual environment into which it installed, and when you are done using Keras, deactivate the environment.

#for python 2

(keras-tf-venv)$ deactivate

$  # Your prompt should change back

#for python 3

(keras-tf-venv3)$ deactivate

$  # Your prompt should change back

For more information : http://deeplearning.lipingyang.org/2017/08/01/install-keras-with-tensorflow-backend/
