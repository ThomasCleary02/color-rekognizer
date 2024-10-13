FROM public.ecr.aws/lambda/python:3.9

# Set the working directory to /var/task
WORKDIR /var/task

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Install system dependencies for OpenCV
RUN yum update -y && yum install -y \
    mesa-libGL \
    && yum clean all

# Copy the rest of your application code
COPY . ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "api.lambda_handler" ]