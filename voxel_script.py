import argparse
import glob
import os
import open3d as o3d
import numpy as np


def Initialize_cube(mesh, voxel_size, bounds):
    
    # alternative method
    # cube = np.mgrid[:voxel_size, :voxel_size, :voxel_size]
    # cube = cube.transpose(1, 2 ,3, 0)
    
    cube = []
    x = np.linspace(bounds[0], bounds[1], voxel_size)
    y = x.copy()
    z = x.copy()
    for xi in x:
        for yi in y:
            for zi in z:
                cube.append([xi, yi, zi])
                
    cube = np.array(cube)
    cube = cube + np.asarray(mesh.get_center())
    
    return cube



def check_occupancy(mesh_path, mesh_number, args):
    
    bounds = [-1, 1]
    diff = bounds[1] - bounds[0]
    
    # mesh to point cloud
    input_mesh = o3d.io.read_triangle_mesh(mesh_path)
    # number_of_points = 2000
    # pcd = input_mesh.sample_points_poisson_disk(number_of_points, init_factor=5, pcl=None)
    
    # normalization
    scale = np.max(input_mesh.get_max_bound() - input_mesh.get_min_bound())
    print('scale:', scale)
    
    # create scale directory
    scale_dir = args.raw_dataset_dir + str(mesh_number) + '/' + args.scale_dir
    os.makedirs(scale_dir, exist_ok=True)
    np.save(scale_dir + str(mesh_number) + '.npy', np.array([scale]))
    
    # saling
    input_mesh.scale(diff / scale, center=input_mesh.get_center())
    
    # include checking
    queries = Initialize_cube(input_mesh, args.voxel_size, bounds)
    
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(input_mesh, voxel_size=diff/args.voxel_size)
    
    """
    # save cube
    cube_pcd = o3d.geometry.PointCloud()
    cube_pcd.points = o3d.utility.Vector3dVector(queries)
    cube_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    cube_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(cube_pcd, depth=8, width=0, scale=1.1, linear_fit=False)
    cube_mesh = cube_mesh[0]
    cube_mesh.paint_uniform_color([1.0, 1.0, 1.0])  # white?
    cube_mesh_dir = args.raw_dataset_dir + str(mesh_number) + '/' + 'CubeMesh/'
    os.makedirs(cube_mesh_dir, exist_ok=True)
    
    o3d.io.write_triangle_mesh(cube_mesh_dir + '{}.obj'.format(mesh_number), cube_mesh)
    """
    
    queries = o3d.utility.Vector3dVector(queries)
    voxel_flag = voxel_grid.check_if_included(queries)
    
    voxel_data = np.array(voxel_flag).reshape(args.voxel_size, args.voxel_size, args.voxel_size)
    
    # create voxel directory
    voxel_dir = args.raw_dataset_dir + str(mesh_number) + '/Voxel{}/'.format(args.voxel_size) 
    
    os.makedirs(voxel_dir, exist_ok=True)
    
    voxel_path = voxel_dir + '{}.npz'.format(mesh_number)
    
    # save at npz format
    np.savez(voxel_path, voxel_data)
    
# Loop main function
def get_occupied_mesh(args):
    object_directories = os.listdir(args.raw_dataset_dir)
    object_numbers = sorted([int(o) for o in object_directories])
    
    for on in object_numbers:
        print()
        print('---progress {} mesh---'.format(on))
        mesh_path = args.raw_dataset_dir + str(on) + '/' + args.ml_mesh_dir + str(on) + '.obj'
        
        voxel_dir = args.raw_dataset_dir + str(on) + '/' + args.voxel_dir + '/'
        
        # if os.path.exists(voxel_dir):
        #    continue
        
        check_occupancy(mesh_path, on, args)
    
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxel_size', type=int, default=64)
    parser.add_argument('--scale_dir', type=str, default='Scale/')
    parser.add_argument('--raw_dataset_dir', type=str, default='./RawDataset/')
    parser.add_argument('--ml_mesh_dir', type=str, default='MLMesh/')
    args = parser.parse_args()
    
    get_occupied_mesh(args)
    
    # test for single mesh
    # mesh_number = 25
    # mesh_path = args.raw_dataset_dir + str(mesh_number) + '/' + args.ml_mesh_dir + '{}.obj'.format(mesh_number)
    # check_occupancy(mesh_path, mesh_number, args)
        

    
if __name__ == '__main__':
    main()