#!/bin/bash

# Function to run a command and check for errors
run_command() {
    echo "Running: $1"
    if ! eval $1; then
        echo "Error executing: $1"
        exit 1
    fi
}

# Clone Darknet repository
echo "Cloning Darknet repository..."
run_command "git clone https://github.com/AlexeyAB/darknet.git"
cd darknet

# Modify Makefile
echo "Modifying Makefile..."
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile

# Ensure CUDNN paths are set correctly
echo "CUDNN_PATH=/usr/local/cuda" >> Makefile

# Compile Darknet
echo "Compiling Darknet..."
run_command "make"

echo "Darknet compiled successfully!"

# Print system information
echo -e "\nSystem Information:"
nvcc --version
gcc --version
uname -a
lsb_release -a

echo -e "\nDarknet setup complete!"