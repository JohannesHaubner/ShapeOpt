#! /usr/bin/bash

pip3 uninstall -y cyipopt
cd /usr/local/
git clone https://github.com/coin-or-tools/ThirdParty-HSL.git
cd ThirdParty-HSL
cp -r /home/haubnerj/shapeopt/hsl/coinhsl .
mkdir build
cd build
../configure --prefix=/usr/local
make 
make install
cd /usr/local/Ipopt
rm -r build
mkdir build
cd build
../configure --without-asl  --with-lapack --without-mumps --with-hsl --prefix=/usr/local
make && make test && make install

cd ../..

pip3 install git+https://github.com/mechmotum/cyipopt.git


