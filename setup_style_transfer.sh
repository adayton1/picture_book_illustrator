#!/usr/bin/env bash

# Tell the user we're starting
echo "Setting up fast style transfer...";

# Ask if they want to download vgg16 weights
read -r -p "Download vgg16 weights? Needed if training. [y/N] " input;
downloadVGG16Weights=false;
case $input in
    [yY][eE][sS]|[yY]) downloadVGG16Weights=true;;
esac

# Create a folder for holding dependencies
localFolder="./deps";
mkdir -p "$localFolder";
cd "$localFolder";

# Clone faststyle repository if needed
faststyleDir="./faststyle";
if [ -d "$faststyleDir" ]; then
    echo "The repository already exists. Not cloning.";
else
    echo "Cloning the faststyle repository...";
    repository="git@github.com:ghwatson/faststyle.git";
    git clone "$repository";
fi

# Check if the clone worked
if [ ! -d "$faststyleDir" ]; then
    echo "Could not clone the repository. Exiting now.";
    exit 1;
fi

# Check if need to download the training data
if [ "$downloadVGG16Weights" = true ]; then
    echo "Downloading vgg16 weights...";
    cd faststyle/libs;
    ./get_vgg16_weights.sh;
fi
