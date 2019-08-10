docker run -dit --runtime=nvidia \
--name 'tianchi2019-pytorch' \
-p '9002:8888' \
-v '/home:/workspace' \
-v '/home/tianchi2019/docker/jupyter_notebook_config.json:/root/.jupyter/jupyter_notebook_config.json' \
pytorch/pytorch:latest