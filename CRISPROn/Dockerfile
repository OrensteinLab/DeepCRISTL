# Use an official Python runtime as a parent image
FROM python:3.6.13
# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install \
    biopython==1.71 \
    h5py==2.10.0 \
    Keras==2.4.3 \
    python-Levenshtein==0.21.1 \
    matplotlib==2.2.0 \
    numpy==1.19.5 \
    pandas==1.1.5 \
    scikit-learn==0.20.0 \
    scipy==1.5.4 \
    tensorflow==2.4.1 \ 
    seaborn==0.11.2 \
    shap \
    logomaker


# Copy the current directory contents into the container at /app
COPY . /app

# Define environment variable
ENV NAME World


# Run script.py when the container launches
ENTRYPOINT  ["python", "tool.py"]
