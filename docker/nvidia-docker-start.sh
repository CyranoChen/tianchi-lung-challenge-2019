docker run -d --runtime=nvidia \
--name 'tianchi2019' \
-p '9001:8888' \
-v '/home:/tf' \
-v '/home/tianchi2019/docker/jupyter_notebook_config.json:/root/.jupyter/jupyter_notebook_config.json' \
tensorflow/tensorflow:latest-gpu-py3-jupyter