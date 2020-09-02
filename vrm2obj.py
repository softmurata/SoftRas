import glob
import os
import subprocess

root_dir = '../../Downloads/VRModel/'  # download raw vrm file

root_vrm_files = glob.glob(root_dir + '*.vrm')
vrm_numbers = list(range(len(root_vrm_files)))

temp_vrm_dir = './VRMMesh/'

os.makedirs(temp_vrm_dir, exist_ok=True)

vrm_files = [temp_vrm_dir + str(n) + '.vrm' for n in vrm_numbers]

temp_glb_dir = './GLBMesh/'

os.makedirs(temp_glb_dir, exist_ok=True)

glb_files = [temp_glb_dir + str(n) + '.glb' for n in vrm_numbers]

# change vrm file name
for root_vrm, current_vrm, glb in zip(root_vrm_files, vrm_files, glb_files):
    args = 'cp -f {} {}'.format(root_vrm, current_vrm)
    subprocess.call(args, shell=True)
    args = 'mv {} {}'.format(current_vrm, glb)
    subprocess.call(args, shell=True)
    
args = 'rm -rf {}'.format(temp_vrm_dir[:-1])
subprocess.call(args, shell=True)










