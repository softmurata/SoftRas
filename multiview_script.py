import argparse
import glob
import open3d as o3d
import os
import numpy as np
import matplotlib.pyplot as plt

# Adjust pose of mesh
def adjust_mesh_pose(angles):
    # angle
    anglex, angley, anglez = angles
    
    # x axis rotation
    anglex = anglex * np.pi / 180.0
    Rx = np.array([[1, 0,              0],
                   [0, np.cos(anglex), -np.sin(anglex)],
                   [0, np.sin(anglex), np.cos(anglex)]])
    
    # y axis rotation
    angley = angley * np.pi / 180.0
    Ry = np.array([[np.cos(angley),  0, np.sin(angley)],
                  [0,               1, 0],
                  [-np.sin(angley), 0, np.cos(angley)]])
    
    # z axis rotation
    anglez = anglez * np.pi / 180.0
    Rz = np.array([[np.cos(anglez), -np.sin(anglez), 0],
                   [np.sin(anglez), np.cos(anglez),  0],
                   [0,              0,               1]])
    
    R = Rz.dot(Rx).dot(Ry)
    
    return R


# Take multiview pictures from mesh
def custom_draw_geometry_with_rotation(geometry, mesh_number, args):
    custom_draw_geometry_with_rotation.index = 0
    custom_draw_geometry_with_rotation.vis = o3d.visualization.Visualizer()
    
    mesh_dir = args.raw_dataset_dir + str(mesh_number) + '/'
    
    # make necessary directories
    image_dir = mesh_dir + args.image_dir
    silhouette_dir = mesh_dir + args.silhouette_dir
    traje_dir = mesh_dir + args.traje_dir
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(silhouette_dir, exist_ok=True)
    os.makedirs(traje_dir, exist_ok=True)
    
    # rotation view function
    def rotation_view(vis):
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_rotation
        
        # vel = 90.0 # 24
        # vel = 30.0 # 72
        # vel = 15.0 # 144
        vel = 6.0  # 360
        
        ctr.rotate(vel, 0.0)  # rotate(x, y, xo, yo)
        
        
        # get rgb and depth image
        depth = vis.capture_depth_float_buffer(False)
        image = vis.capture_screen_float_buffer(False)
        
        # convert numpy format
        num_depth = np.asarray(depth)
        num_image = np.asarray(image)
        
        # extract silhouette
        sil_flag = num_depth > 0
        silhouette = np.zeros_like(num_depth)  # Initialize silhouette array
        silhouette[:, :] = 255
        sil_image = np.multiply(silhouette, sil_flag)
        
        
        # save figure(confirmation)
        # {:05d} => 00000
        plt.imsave(image_dir + '{}.png'.format(glb.index), num_image, dpi=1)
        plt.imsave(silhouette_dir + '{}.png'.format(glb.index), sil_image, dpi=1)
        
        # save camera matrix
        camera_params = ctr.convert_to_pinhole_camera_parameters()
        np.save(traje_dir + 'extrinsic{}.npy'.format(glb.index), camera_params.extrinsic)
        np.save(traje_dir + 'intrinsic{}.npy'.format(glb.index), camera_params.intrinsic.intrinsic_matrix)
        
        glb.index += 1
        
        if glb.index == args.multiview:
            exit()
            
        return False
    
    width = 1024
    height = 1024
    window_name='Mesh{}'.format(mesh_number)
    
    vis = custom_draw_geometry_with_rotation.vis
    vis.create_window(window_name=window_name)
    vis.add_geometry(geometry)
    vis.register_animation_callback(rotation_view)
    vis.run()
    vis.destroy_window()
    
    
    
    
    
    
# create multiview figure
def multiview_single(mesh_path, mesh_number, args):
    mesh_dir = args.raw_dataset_dir + str(mesh_number) + '/' + args.ml_mesh_dir
    
    os.makedirs(mesh_dir, exist_ok=True)
    
    # load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # get mesh center
    mesh_center = mesh.get_center()
    
    # search sets of frontal angles
    angles = [float(a) for a in args.angles.split('/')]
    R = adjust_mesh_pose(angles)
    
    # rotate mesh
    mesh.rotate(R, mesh_center)
    
    # paint color
    mesh.paint_uniform_color([1, 0.5, 0])
    
    # compute normals => it is important to describe mesh in real
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    # save output mesh directory
    o3d.io.write_triangle_mesh(mesh_dir + '{}.obj'.format(mesh_number), mesh)
    
    # take multiview pictures
    geometry = mesh
    custom_draw_geometry_with_rotation(geometry, mesh_number, args)
    
    
def take_multiview_pictures_from_mesh(args):
    # mesh files
    mesh_obj_files = os.listdir(args.base_mesh_dir)
    mesh_numbers = [int(mo.split('.')[0]) for mo in mesh_obj_files]
    sorted_index = np.argsort(mesh_numbers)
    
    mesh_numbers = [mesh_numbers[idx] for idx in sorted_index]
    mesh_obj_files = [args.base_mesh_dir + '{}.obj'.format(num) for num in mesh_numbers]
    mesh_mtl_files = [args.base_mesh_dir + '{}.mtl'.format(num) for num in mesh_numbers]
    
    
    for mesh_path, mesh_number in zip(mesh_obj_files, mesh_numbers):
        print()
        print('----make progress {} process----'.format(mesh_number))
        multiview_single(mesh_path, mesh_number, args)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, default='images/')  # rgb image directory
    parser.add_argument('--silhouette_dir', type=str, default='silhouette/')  # silhouette directory
    parser.add_argument('--traje_dir', type=str, default='cameramat/')  # camera matrix directory
    parser.add_argument('--base_mesh_dir', type=str, default='BaseMesh/')  # target mesh before rotation
    parser.add_argument('--raw_dataset_dir', type=str, default='./RawDataset144/')  
    parser.add_argument('--ml_mesh_dir', type=str, default='MLMesh/')  # target mesh after rotation(for machine learning)
    parser.add_argument('--angles', type=str, default='0/180/0', help='x axis angle/ y axis angle/ z axis angle')
    parser.add_argument('--multiview', type=int, default=360)
    parser.add_argument('--mesh_number', type=int, default=10000)
    args = parser.parse_args()
    
    # take multiview pictures(Main function)
    # take_multiview_pictures_from_mesh(args)
    
    # test in order to check taking multiview pictures from single mesh correctly
    mesh_number = args.mesh_number
    mesh_path = args.base_mesh_dir + '{}.obj'.format(mesh_number)
    multiview_single(mesh_path, mesh_number, args)
    
    
    
if __name__ == '__main__':
    main()
    
    
    