
# We found a python script for parameterization - let's read and parse parameter values from it
with open('output/seicbrsbmschoolparameterization.py', 'r') as f:
    param_script = f.read()

param_script[:1000]  # Show first 1000 characters of script for inspection