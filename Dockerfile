FROM --platform=linux/amd64 pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Add the directory containing the scripts to PATH
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY --chown=user:user ./ /opt/app/
COPY --chown=user:user _model /opt/app/_model


RUN python -m pip install \
    --user \
    -e .

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inferenceTask2.py"]