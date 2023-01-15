# Use github pages for docker image
FROM ghcr.io/johanneshaubner/shapeopt:latest

# Create user with a home directory
ARG NB_USER
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV HOME /home/${NB_USER}

# Copy current directory
WORKDIR ${HOME}
COPY . ${HOME}

# Change ownership of home directory
USER root
RUN chown -R ${NB_UID} ${HOME}

USER ${NB_USER}
ENTRYPOINT []

RUN /bin/bash -l -c "pip3 uninstall -y cyipopt"
RUN cd /usr/local &&\
    git clone https://github.com/coin-or-tools/ThirdParty-HSL.git &&\
    cd ThirdParty-HSL &&\
    cp -r ${HOME]/shapeopt/hsl/coinhsl . &&\
    mkdir build && cd build && ../configure --prefix=/usr/local &&\
    make && make install &&\
    cd /usr/local/Ipopt && rm -r build && mkdir build && cd build &&\
    ../configure --without-asl --without-mumps --with-lapack --with-hsl --prefix=/usr/local &&\
    make && make test && make install
RUN /bin/bash -l -c "pip3 install git+https://github.com/mechmotum/cyipopt.git"