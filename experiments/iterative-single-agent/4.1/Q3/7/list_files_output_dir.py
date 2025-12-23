
# Check file list in output directory to confirm what files exist
import os
output_dir = os.path.join(os.getcwd(), 'output')
file_list = os.listdir(output_dir)
file_list