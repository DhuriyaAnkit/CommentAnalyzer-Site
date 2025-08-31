#!/bin/bash
# Use a specific Python version to install packages
python3.9 -m pip install -r requirements.txt
# Use the same Python version to run Django commands
python3.9 manage.py collectstatic --noinput 
python3.9 manage.py migrate