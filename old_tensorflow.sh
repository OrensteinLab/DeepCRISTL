#!/bin/bash

# Define an associative array to store packages and versions
declare -A packages=(
    ["h5py"]="2.7.1"
    ["keras"]="2.1.6"
    ["tensorflow"]="1.8.0"
    # Add more packages as needed
)

# Iterate over the array and install each package with the specified version
for package in "${!packages[@]}"; do
    version=${packages[${package}]}

    echo "Installing ${package} version ${version}..."
    pip install --force-reinstall ${package}==${version}

    # Check if the installation was successful
    if [ $? -eq 0 ]; then
        echo "Installation of ${package} version ${version} successful."
    else
        echo "Installation of ${package} version ${version} failed. Please check the error messages above."
    fi
done
