FROM public.ecr.aws/lambda/python:3.11


RUN dnf install -y \
    gdal \
    gdal-devel \
    && dnf clean all


COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY src/ ${LAMBDA_TASK_ROOT}/
COPY data/ ${LAMBDA_TASK_ROOT}/data/


CMD ["inference.handler.lambda_handler"]
