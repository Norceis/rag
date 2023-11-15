# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src /app/src
COPY requirements_docker.txt /app/requirements.txt
COPY .env /app/.env
VOLUME /app/data
VOLUME /app/models
VOLUME /app/notebooks
VOLUME /app/src

# Define environment variable
ENV DEBIAN_FRONTEND=noninteractive
ENV CMAKE_ARGS "-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE 1
RUN apt-get update && apt-get install -y gcc clang clang-tools cmake python3

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8888 available to the world outside this container
EXPOSE 8888

# Command to run on container start
#CMD [ "streamlit", "run", "src/app/main.py", "--server.port=8888", "--server.address=0.0.0.0"]

#cd src/app; streamlit run main.py --server.port=8888 --server.address=0.0.0.0
#jupyter notebook --ip 0.0.0.0 --port 8888 --allow-root
#docker run `
#  -v C:\Users\cubix\PycharmProjects\rag\data:/app/data `
#  -v C:\Users\cubix\PycharmProjects\rag\models:/app/models `
#  -v C:\Users\cubix\PycharmProjects\rag\notebooks:/app/notebooks `
#  -v C:\Users\cubix\PycharmProjects\rag\src:/app/src `
#  -p 8888:8888 `
#  --gpus all `
#  -it `
#  --name test-rag `
#  test-rag