# create mesh data by blender-python ( Downloads/blender_python.py )
# 1. convert vrm into obj
# take multiview pictures from mesh ( Documents/MaskFusion/Open3d/get_depth_from_mesh.py )
# 1. get rgb data and depth data
# 2. convert depth data into silhouette data
# save rgb + silhouette data and create voxel data(True and False format)

# PIFu
# create normal map

import argparse
from multiview_script import *
from voxel_script import *
from image_script import *


def main():
    parser = argparse.ArgumentParser()
    # for multiview
    parser.add_argument('--raw_dataset_dir', type=str, default='./RawDataset/')
    parser.add_argument('--image_dir', type=str, default='images/')  # rgb image directory
    parser.add_argument('--silhouette_dir', type=str, default='silhouette/')  # silhouette directory
    parser.add_argument('--traje_dir', type=str, default='cameramat/')  # camera matrix directory
    parser.add_argument('--base_mesh_dir', type=str, default='BaseMesh/')  # target mesh before rotation
    parser.add_argument('--ml_mesh_dir', type=str, default='MLMesh/')  # target mesh after rotation
    parser.add_argument('--angles', type=str, default='0/180/0', help='x axis angle/ y axis angle/ z axis angle')
    parser.add_argument('--multiview', type=int, default=24)
    # for voxel
    parser.add_argument('--voxel_size', type=int, default=32)
    parser.add_argument('--voxel_dir', type=str, default='Voxel/')
    parser.add_argument('--scale_dir', type=str, default='Scale/')
    # for image
    parser.add_argument('--ml_height', type=int, default=64)
    parser.add_argument('--ml_width', type=int, default=64)
    parser.add_argument('--mldata_dir', type=str, default='mldataset/')
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--class_name', type=str, default='character')
    args = parser.parse_args()
    
    # multiview
    take_multiview_pictures_from_mesh(args)
    # voxel
    get_occupied_mesh(args)
    # image
    set_data(args)
    
    
if __name__ == '__main__':
    main()
    


