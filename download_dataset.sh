#!/bin/bash

# Configure AWS credentials if not already set.
if ! aws configure get aws_access_key_id > /dev/null 2>&1; then
    echo "AWS credentials not found. Running aws configure..."
    aws configure
fi

# Download to the local directory the folder you wish to train with.
echo "Select an option: 1 - Download uncleaned raw dataset, 2 - Download cleaned dataset, 3 - Skip download"
read -p "Enter your choice (1, 2, or 3): " choice

case $choice in
    1)
        echo "Downloading uncleaned raw dataset..."
        mkdir -p archive
        aws s3 cp s3://resistor-classifier/archive ./archive --recursive
        ;;
    2)
        echo "Downloading cleaned dataset..."
        mkdir -p archive_clean
        aws s3 cp s3://resistor-classifier/archive_clean ./archive_clean --recursive
        ;;
    3)
        exit 0
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1, 2, or 3."
        exit 1
        ;;
esac