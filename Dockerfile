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