import os

import soft_renderer.functional as srf
import torch
import numpy as np
import tqdm


# Mono class
class_id_map = {'0': 'character'}

class MonoShapeNet(object):

    def __init__(self, args, directory=None, class_id=None, set_name=None):

        self.class_id = class_id
        self.set_name = set_name  # train or val
        self.elevation = 30.0
        self.distance = 2.732

        self.class_id_map = class_id_map

        self.image_size = args.image_size
        self.voxel_size = args.voxel_size
        self.num_views = args.view_point
        self.view_angle = 360 / self.num_views

        images = []
        voxels = []

        self.num_data = {}  # the number of data
        self.pos = {}  # start position
        count = 0
        loop = tqdm.tqdm(self.class_id)
        loop.set_description('Loading dataset')

        for class_id in loop:
            # load image data(RGBD) from image npz file
            images.append(list(np.load(os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items())[0][1])

            voxels.append(list(np.load(os.path.join(directory, '%s_%s_voxels.npz' % (class_id, set_name))).items())[0][1])

            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count

            count += self.num_data[class_id]

        # change images shape
        images = np.concatenate(images, axis=0).reshape((-1, 4, self.image_size, self.image_size))  # base images dataset shape = (num_pictures, multiview_points, 4, image_height, image_width)
        images = np.ascontiguousarray(images)
        self.images = images

        voxels = np.concatenate(voxels, axis=0)
        voxels = np.ascontiguousarray(voxels)
        self.voxels = voxels

        # diminish the volume of CPU
        del images
        del voxels

    @property
    def class_ids_pair(self):
        class_name = [self.class_id_map[i] for i in self.class_id]
        return zip(self.class_id, class_name)

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = torch.zeros(batch_size)
        viewpoint_ids_b = torch.zeros(batch_size)
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            viewpoint_id_a = np.random.randint(0, self.view_point)
            viewpoint_id_b = np.random.randint(0, self.view_point)
            data_id_a = (object_id + self.pos[class_id]) * self.view_point + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * self.view_point + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = torch.from_numpy(self.images[data_ids_a].astype('float32') / 255.)
        images_b = torch.from_numpy(self.images[data_ids_b].astype('float32') / 255.)

        distances = torch.ones(batch_size).float() * self.distance
        elevations_a = torch.ones(batch_size).float() * self.elevation
        elevations_b = torch.ones(batch_size).float() * self.elevation
        viewpoints_a = srf.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * self.view_angle)
        viewpoints_b = srf.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * self.view_angle)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(self.num_views), data_ids.size)
        data_ids = np.repeat(data_ids, self.num_views) * self.num_views + viewpoint_ids

        distances = torch.ones(data_ids.size).float() * self.distance
        elevations = torch.ones(data_ids.size).float() * self.elevation
        viewpoints_all = srf.get_points_from_angles(distances, elevations, -torch.from_numpy(viewpoint_ids).float() * self.view_angle)

        for i in range((data_ids.size - 1) // batch_size + 1):
            images = torch.from_numpy(self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.)
            voxels = torch.from_numpy(self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] // self.num_views].astype('float32'))
            yield images, voxels

    def get_single_image(self, class_id, mesh_id, view_id):
        data_id = self.pos[class_id] + mesh_id * self.num_views + view_id
        print('data_id:', data_id)
        images = torch.from_numpy(self.images[data_id].astype('float32') / 255.0)
        voxels = torch.from_numpy(self.voxels[data_id // self.num_views].astype('float32'))

        images = images.view(1, 4, self.image_size, self.image_size)
        voxels = voxels.view(1, self.voxel_size, self.voxel_size, self.voxel_size)

        return images, voxels
