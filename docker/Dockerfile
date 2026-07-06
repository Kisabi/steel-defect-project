FROM rocm/pytorch:rocm7.2.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
# TODO: доустановить зависимости проекта
COPY requirements.txt /tmp/requirements.txt
RUN pip install --break-system-packages -r /tmp/requirements.txt
WORKDIR /workspace
