#xeongold16q

module load gcc
module load cmake/gcc/3.20.1
module load intel/mpi
module load intel/mkl/64/2020/4.304


#create folder $WORK/install_files_2017 and put all tarballs there


#make folder
cd $HOME
mkdir FEniCS21
cd FEniCS21

wget ftp://sourceware.org/pub/libffi/libffi-3.2.1.tar.gz
tar xzf libffi-3.2.1.tar.gz
cd libffi-3.2.1
./configure --disable-docs --prefix=$HOME/FEniCS21/installed 
make
make install
export LD_LIBRARY_PATH=$HOME/FEniCS21/installed/lib:${LD_LIBRARY_PATH}
export PATH=$HOME/FEniCS21/installed/bin:${PATH}

#install python
cd $HOME/installation_files
tar -xf Python-3.9* -C $HOME/FEniCS21/
cd ../FEniCS21/Python-3.9.6
./configure --enable-shared --with-cc=mpicc --with-cxx=mpicxx \
--with-fc=mpif90 \
 CFLAGS='-O3 -fPIC' \
CXXFLAGS='-O3 -fPIC' \
FFLAGS='-O3 -fPIC' --prefix=$HOME/FEniCS21/Python-3.9.6/installed LDFLAGS="-L/home/haubnerj/FEniCS21/installed/lib" CPPFLAGS="-I /home/haubnerj/FEniCS21/installed/lib/libffi-3.2.1/include"
make
make install
export LD_LIBRARY_PATH=$HOME/FEniCS21/Python-3.9.6/installed/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=$HOME/FEniCS21/Python-3.9.6/installed/lib/python3.9/site-packages
export PATH=$HOME/FEniCS21/Python-3.9.6/installed/bin:${PATH}
export PKG_CONFIG_PATH=$HOME/FEniCS21/installed/lib/pkgconfig/

## install ipopt  HSL #TODO
tar --gzip -xf $HOME/installation_files/IPOPT_INSTALL/Ipopt-3.12.6.tgz -C $HOME/FEniCS21
cd $HOME/FEniCS21/Ipopt-3.12.6/ThirdParty
cp -r $HOME/installation_files/IPOPT_INSTALL/HSL .
cd Metis
./get.Metis
cd ../..
#get.Blas get.Lapack
./configure --prefix=$HOME/FEniCS21/installed
make 
make install
export LD_LIBRARY_PATH=$HOME/FEniCS21/installed/lib:${LD_LIBRARY_PATH}
export PATH=$HOME/FEniCS21/installed/bin:${PATH}
cd ./Ipopt/src/Interfaces/.libs/
ln -s libipopt.so.1 libipopt.so.3

## install cyipopt

pip3 install git+https://github.com/mechmotum/cyipopt.git


#install bzip2
cd $HOME/installation_files
tar --gzip -xf bzip2-latest.tar.gz -C $HOME/FEniCS21/
cd ../FEniCS21/bzip2-1.0.8
export CFLAGS="-fPIC"
export CXXFLAGS="-fPIC"
make -f Makefile-libbz2_so
make
make install PREFIX=$HOME/FEniCS21/bzip2-1.0.8/installed
cp libbz2.so.1.0.8 $HOME/FEniCS21/bzip2-1.0.8/installed/lib/
cd $HOME/FEniCS21/bzip2-1.0.8/installed/lib
ln -s libbz2.so.1.0.8 libbz2.so.1.0
export PATH=$HOME/FEniCS21/bzip2-1.0.8/installed/bin:${PATH}
export LD_LIBRARY_PATH=$HOME/FEniCS21/bzip2-1.0.8/installed/lib:${LD_LIBRARY_PATH}

#install boost
cd $HOME/installation_files
tar --gzip -xf boost_1_74_0.tar.gz -C $HOME/FEniCS21/
cd ../FEniCS21/boost_1_74_0/tools/build
./bootstrap.sh 
./b2 install --prefix=$HOME/FEniCS21/boost_1_74_0/installed 
cd ../..
export PATH=$HOME/FEniCS21/boost_1_74_0/installed/bin:${PATH}
b2 --build-dir=$HOME/FEniCS21/boost_1_74_0/installed toolset=gcc stage
export LD_LIBRARY_PATH=$HOME/FEniCS21/boost_1_74_0/stage/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$HOME/FEniCS21/boost_1_74_0/libs:${LD_LIBRARY_PATH}


#install pcre
cd $HOME/installation_files
tar --gzip -xf pcre2-10.37.tar.gz -C $HOME/FEniCS21/
cd ../FEniCS21/pcre2-10.37
./configure CC=gcc CFLAGS=-O3 CXX=g++ CXXFLAGS=-O3 --prefix=$HOME/FEniCS21/installed
make
make install
export LD_LIBRARY_PATH=$HOME/FEniCS21/installed/lib:${LD_LIBRARY_PATH}
export PATH=$HOME/FEniCS21/installed/bin:${PATH}

#install swig

cd $HOME/installation_files
tar --gzip -xf swig-4.0.2.tar.gz -C $HOME/FEniCS21/
cd ../FEniCS21/swig-4.0.2
./configure CC=gcc CFLAGS=-O3 CXX=g++ CXXFLAGS=-O3 --prefix=$HOME/FEniCS21/installed
make 
make install

#install VTK
cd $HOME/installation_files
tar --gzip -xf VTK-9.0.3.tar.gz -C $HOME/FEniCS21/
cd ../FEniCS21/VTK-9.0.3/
mkdir build
cd build
cmake -DPYTHON_INCLUDE_DIR=/home/haubnerj/FEniCS21/Python-3.9.6/  \
             -DBUILD_SHARED_LIBS:BOOL=ON -DVTK_WRAP_PYTHON=ON                              \
            -DVTK_PYTHON_VERSION=3 -DCMAKE_INSTALL_PREFIX:PATH=$HOME/FEniCS21/installed -DCMAKE_BUILD_TYPE=Release ..
make
make install

python3 --version

#install HDF5 1.10.6
cd $HOME/installation_files
tar --gzip -xf hdf5-1.10.6.tar.gz -C $HOME/FEniCS21/
cd $HOME/FEniCS21/hdf5-1.10.6
./configure --enable-parallel --enable-fortran --enable-shared  --with-cc=mpicc --with-cxx=mpicxx \
--with-fc=mpif90 --prefix=$HOME/FEniCS21/installed \
 CFLAGS='-O3 -fPIC' \
CXXFLAGS='-O3 -fPIC' \
FFLAGS='-O3 -fPIC'
make 
make install


#install PETSC
cd $HOME/installation_files
tar --gzip -xf petsc-3.14.0.tar.gz -C $HOME/FEniCS21/
cd $HOME/FEniCS21/petsc-3.14.0/
./configure --enable-shared --prefix=$HOME/FEniCS21/petsc-3.14.0/installed \
 --with-cc=mpicc --with-cxx=mpicxx \
--with-fc=mpif90 --with-hdf5-dir=/home/haubnerj/FEniCS21/installed \
--download-scotch=yes --download-parmetis=yes \
--download-metis=yes --download-suitesparse=yes --with-debugging=0 \
--download-mumps=yes --download-scalapack=yes --download-superlu_dist=yes --download-ptscotch=yes --download-superlu=yes --download-ml=yes --download-fftw=yes --download-hypre=yes \
 CFLAGS='-O3 -fPIC' \
CXXFLAGS='-O3 -fPIC' \
FFLAGS='-O3 -fPIC'

make PETSC_DIR=/home/haubnerj/FEniCS21/petsc-3.14.0 PETSC_ARCH=arch-linux-c-opt all
make PETSC_DIR=/home/haubnerj/FEniCS21/petsc-3.14.0 PETSC_ARCH=arch-linux-c-opt install
make PETSC_DIR=/home/haubnerj/FEniCS21/petsc-3.14.0/installed PETSC_ARCH="" check
export LD_LIBRARY_PATH=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/lib:${LD_LIBRARY_PATH}
export PATH=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/bin:${PATH}
export PETSC_DIR=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/
export PETSC_ARCH=""
cd ..

#install eigen
cd $HOME/installation_files
tar --gzip -xf eigen-3.3.9.tar.gz -C $HOME/FEniCS21/
cd ../FEniCS21/eigen-3.3.9
mkdir build
cd build
cmake -DBOOST_ROOT:PATH=$HOME/FEniCS21/boost_1_74_0 -DCMAKE_C_COMPILER=mpicc -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_FC_COMPILER=mpif90 \
CFLAGS='-O3 -fPIC' CXXFLAGS=' -O3 -fPIC' FFLAGS='-O3 -fPIC' -DCMAKE_INSTALL_PREFIX=$HOME/FEniCS21/installed -DCMAKE_BUILD_TYPE=Release ..
make install

#install slepc
cd $HOME/installation_files
tar --gzip -xf slepc-3.14.0.tar.gz -C $HOME/FEniCS21/
cd ../FEniCS21/slepc-3.14.0
./configure --prefix=$HOME/FEniCS21/installed 
make SLEPC_DIR=/home/haubnerj/FEniCS21/slepc-3.14.0 PETSC_DIR=/home/haubnerj/FEniCS21/petsc-3.14.0/installed
make SLEPC_DIR=/home/haubnerj/FEniCS21/slepc-3.14.0 PETSC_DIR=/home/haubnerj/FEniCS21/petsc-3.14.0/installed install

export SLEPC_DIR=/home/haubnerj/FEniCS21/installed 

#install python packages

##other packages analogously: ply, petsc4py, mpmath, sympy, six, ufl-master, fiat-master, instant-master, ffc-master
#export PYTHONPATH=$WORK/FEniCS21_2_2016/python_packages/installed/lib/python2.7/site-packages:$WORK/#FEniCS21_2_2016/python_packages/installed/lib/python2.7/site-packages:${PYTHONPATH}
#export LD_LIBRARY_PATH=$WORK/FEniCS21_2_2016/python_packages/installed/lib:$WORK/FEniCS21_2_2016/#python_packages/installed/lib:${LD_LIBRARY_PATH}


#cd $WORK/FEniCS21_2_2016#
#mkdir python_packages
#cd python_packages

#module load git


## install numpy and scipy

# install pybind version > 2.6.0  /home/haubnerj/FEniCS21/Python-3.9.6/installed/lib/python3.9/site-packages/pybind11-2.6.2-py3.9.egg/pybind11/share/cmake/pybind11 !!!

cd $HOME/installation_files
tar --gzip -xf pybind.tar.gz -C $HOME/FEniCS21
cd $HOME/FEniCS21/pybind-pybind11-c7faa0f/
python3 setup.py install

pip3 install numpy
pip3 install scipy
pip3 install Sphinx

cd $HOME/installation_files
tar --gzip -xf petsc4py-3.14.0.tar.gz -C $HOME/FEniCS21
tar --gzip -xf slepc4py-3.14.0.tar.gz -C $HOME/FEniCS21
cd $HOME/FEniCS21/petsc4py-3.14.0
python3 setup.py install
cd ../slepc4py-3.14.0
python3 setup.py install

cd $HOME/FEniCS21

## rest	

cd $HOME/FEniCS21
git clone https://bitbucket.org/FEniCS-project/fiat.git
git clone https://bitbucket.org/FEniCS-project/dijitso.git
git clone https://bitbucket.org/FEniCS-project/ufl.git
git clone https://bitbucket.org/FEniCS-project/ffc.git
git clone https://bitbucket.org/FEniCS-project/dolfin.git
git clone https://bitbucket.org/FEniCS-project/mshr.git
cd fiat    && pip3 install .
cd ..
cd dijitso && pip3 install .
cd ..
cd ufl     && pip3 install .
cd ..
cd ffc     && pip3 install .
cd ..
# apply patches to dolfin (patches from anaconda installation)
cd dolfin  && mkdir build && cd build 
cmake -DBOOST_ROOT:PATH=$HOME/FEniCS21/boost_1_74_0 \
-DDOLFIN_SKIP_BUILD_TESTS=True \
-DBOOST_ROOT:PATH=$HOME/FEniCS21/boost_1_74_0 -DBOOST_HOME:PATH=$HOME/FEniCS21/boost_1_74_0 -DBOOST_LIBRARYDIR:PATH=$HOME/FEniCS21/boost_1_74_0/stage/lib \
-DSLEPC_DIR:PATH=$WORK/FEniCS21/installed \
-DSWIG_DIR:PATH=$WORK/FEniCS21/installed \
-DCMAKE_INSTALL_PREFIX=$HOME/FEniCS21/installed -DDOLFIN_ENABLE_TRILINOS=False \
-DDOLFIN_ENABLE_PETSC=True \
-DPETSC_INCLUDE_DIRS=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/include \
-DPETSC_LIBRARY_DIRS=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/lib \
-DCMAKE_BUILD_TYPE=Release ..
make 
make install
cd ../python
python3 setup.py install
# /home/haubnerj/FEniCS21/Python-3.9.6/installed/lib/python3.9/site-packages/pybind11-2.6.2-py3.9.egg/pybind11/share/cmake/pybind11/ in CMakeList.txt
# HINTS /home/haubnerj/FEniCS21/boost_1_74_0/stage/lib/cmake/Boost-1.74.0/ in UseDolfin

pip3 install six
pip3 install matplotlib
pip3 install meshio

## dolfin adjoint

pip3 install git+https://github.com/dolfin-adjoint/pyadjoint.git@master

##set all
module load gcc
module load cmake/gcc/3.20.1
module load intel/mpi
module load intel/mkl/64/2020/4.304

export PKG_CONFIG_PATH=$HOME/FEniCS21/installed/lib/pkgconfig/
export LD_LIBRARY_PATH=$HOME/FEniCS21/Python-3.9.6/installed/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=$HOME/FEniCS21/Python-3.9.6/installed/lib/python3.9/site-packages
export PATH=$HOME/FEniCS21/Python-3.9.6/installed/bin:${PATH}
export PATH=$HOME/FEniCS21/bzip2-1.0.8/installed/bin:${PATH}
export LD_LIBRARY_PATH=$HOME/FEniCS21/bzip2-1.0.8/installed/lib:${LD_LIBRARY_PATH}
export PATH=$HOME/FEniCS21/boost_1_74_0/installed/bin:${PATH}
export LD_LIBRARY_PATH=$HOME/FEniCS21/boost_1_74_0/stage/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$HOME/FEniCS21/boost_1_74_0/libs:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$HOME/FEniCS21/installed/lib:${LD_LIBRARY_PATH}
export PATH=$HOME/FEniCS21/installed/bin:${PATH}
export LD_LIBRARY_PATH=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/lib:${LD_LIBRARY_PATH}
export PATH=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/bin:${PATH}
export PETSC_DIR=/home/haubnerj/FEniCS21/petsc-3.14.0/installed/
export PETSC_ARCH=""
export SLEPC_DIR=$HOME/FEniCS21/installed
export PATH=$HOME/FEniCS21/Python-3.9.6/installed/lib/python3.9/site-packages:${PATH}
export PATH=/cm/shared/apps/intel/compilers_and_libraries/2020.4.304/linux/mkl:${PATH}
export LD_LIBRARY_PATH=/cm/shared/apps/intel/compilers_and_libraries/2020.4.304/linux/mkl/lib:${LD_LIBRARY_PATH}


source /home/haubnerj/FEniCS21/installed/share/dolfin/dolfin.conf





