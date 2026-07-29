FROM public.ecr.aws/lambda/python:3.11

# Install GDAL/rasterio system dependencies
RUN dnf install -y \
    gdal \
    gdal-devel \
    && dnf clean all

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ${LAMBDA_TASK_ROOT}/
COPY data/ ${LAMBDA_TASK_ROOT}/data/

# Lambda handler entrypoint
CMD ["inference.handler.lambda_handler"]
