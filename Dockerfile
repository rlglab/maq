FROM docker.io/pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt update && \
    apt install -y git vim tmux zip build-essential xvfb libosmesa6-dev curl


RUN pip install git+https://github.com/Farama-Foundation/d4rl@71a9549f2091accff93eeff68f1f3ab2c0e0a288
    
RUN mkdir -p /root/.mujoco 
COPY mujoco210.zip /root/.mujoco/mujoco210.zip
RUN unzip /root/.mujoco/mujoco210.zip -d /root/.mujoco


RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin' >> /root/.bashrc
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210

RUN pip install "cython<3" moviepy tensorboard stable-baselines3[extra]==1.8.0 pyrallis==0.3.1


# RUN pip3 install gym[atari]
# RUN DEBIAN_FRONTEND=noninteractive apt install -y libglib2.0-0 libsm6 libxrender1 libxext6
# RUN pip3 install opencv-python 
# RUN apt-get install swig -y
RUN pip3 install envpool
RUN pip3 install "numpy<2"
RUN pip3 install fastdtw
RUN pip3 install scipy \
    scikit-learn \
    POT


WORKDIR /workspace