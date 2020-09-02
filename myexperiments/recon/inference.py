import argparse
import os
import imageio
import numpy as np
import time
import cv2


import torch
import datasets
from utils import AverageMeter, img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import models

"""
# Test code for basic Airplane
BATCH_SIZE = 1
IMAGE_SIZE = 64
CLASS_IDS_ALL = ('02691156')

PRINT_FREQ = 1
SAVE_FREQ = 1

MODEL_DIRECTORY = './data/models'
DATASET_DIRECTORY = './data/datasets'

SIGMA_VAL = 0.01
MESH_ID = 60
VIEW_ID = 0  #(0-23)

parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment_id', type=str)
parser.add_argument('-d', '--model_directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset_directory', type=str, default=DATASET_DIRECTORY)

parser.add_argument('-cls', '--class_ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image_size', type=int, default=IMAGE_SIZE)
parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('-mi', '--mesh_id', type=int, default=MESH_ID)
parser.add_argument('-vi', '--view_id', type=int, default=VIEW_ID)

parser.add_argument('-sv', '--sigma_val', type=float, default=SIGMA_VAL)

parser.add_argument('-pf', '--print_freq', type=int, default=PRINT_FREQ)
parser.add_argument('-sf', '--save_freq', type=int, default=SAVE_FREQ)
args = parser.parse_args()

model = models.Model('data/obj.sphere/sphere_642.obj', args=args)
model = model.cuda()

state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=False)
model.eval()

# dataset loader class
dataset_val = datasets.MonoShapeNet(args.dataset_directory, args.class_ids.split(','), 'val')

directory_output = './data/results/test'
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, args.experiment_id)
os.makedirs(directory_mesh, exist_ok=True)

def test():
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()

    iou_all = []

    for class_id, class_name in dataset_val.class_ids_pair:

        directory_mesh_cls = os.path.join(directory_mesh, class_id)
        os.makedirs(directory_mesh_cls, exist_ok=True)

        images, voxels = dataset_val.get_single_image(class_id, args.mesh_id, args.view_id)

        images = torch.autograd.Variable(images).cuda()  # (batch_size, 4, image_size, image_size)
        voxels = voxels.numpy()

        batch_iou, vertices, faces = model(images, voxels=voxels, task='test')

        batch_time.update(time.time() - end)
        end = time.time()

        # save demo images
        # maybe vertices.size(0) ==> batch_size
        mesh_path = os.path.join(directory_mesh_cls, '%06d.obj' % args.mesh_id)
        input_path = os.path.join(directory_mesh_cls, '%06d.png' % args.mesh_id)
        srf.save_obj(mesh_path, vertices[0], faces[0])  # create mesh
        imageio.imsave(input_path, img_cvt(images[0]))  # create video

        # print loss
        print('Iter: [{0}/{1}]\t'
            'Time {batch_time.val:.3f}\t'
            'IoU {2:.3f}\t'.format(0, ((dataset_val.num_data[class_id] * 24) // args.batch_size),
                                        batch_iou.mean(),
                                        batch_time=batch_time))



test()
"""


BATCH_SIZE = 1
IMAGE_SIZE = 64
CLASS_IDS_ALL = ('0')

PRINT_FREQ = 1
SAVE_FREQ = 1

MODEL_DIRECTORY = './mlresults/models' # => model_directory='models/checkpoint_020000.pth.tar'
DATASET_DIRECTORY = './mldataset'

SIGMA_VAL = 0.01
MESH_ID = 60
VIEW_ID = 0  #(0-23)
RGB_IMAGE_PATH = './RawDataset/{}/images/images/{}.png'.format(MESH_ID, VIEW_ID)
SIL_IMAGE_PATH = './RawDataset/{}/silhouette/images/{}.png'.format(MESH_ID, VIEW_ID)
VOXEL_PATH = './RawDataset/{}/Voxel/{}.npz'.format(MESH_ID, MESH_ID)

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment_id', type=str)
parser.add_argument('-d', '--model_directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset_directory', type=str, default=DATASET_DIRECTORY)

parser.add_argument('-cls', '--class_ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image_size', type=int, default=IMAGE_SIZE)
parser.add_argument('-bs', '--batch_size', type=int, default=BATCH_SIZE)
parser.add_argument('-rimg', '--rgb_image_path', type=str, default=RGB_IMAGE_PATH)
pasrer.add_argument('-simg', '--sil_image_path', type=str, default=SIL_IMAGE_PATH)
parser.add_argument('-vx', '--voxel_path', type=str, default=VOXEL_PATH)

parser.add_argument('-sv', '--sigma_val', type=float, default=SIGMA_VAL)

parser.add_argument('-pf', '--print_freq', type=int, default=PRINT_FREQ)
parser.add_argument('-sf', '--save_freq', type=int, default=SAVE_FREQ)
args = parser.parse_args()


# setup model & optimizer
model = models.Model('mldataset/templatemesh/sphere/sphere_642.obj', args=args)
model = model.cuda()  # for cuda

state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=False)
model.eval()  # switch inference mode

# create rgb image and silhouette
rgb = cv2.cvtColor(cv2.imread(args.rgb_image_path), cv2.COLOR_BGR2RGB)
sil = cv2.cvtColor(cv2.imread(args.sil_image_path), cv2.COLOR_BGR2GRAY)

rgb = rgb.transpose(2, 0, 1)  # (3, height, width)
sil = sil[np.newaxis, :, :]  # (1, height, width)
rgbs = np.concatenate([rgb, sil], axis=0) # (4, height, width)

# Output
directory_output = './mlresults/test'
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, args.experiment_id)
os.makedirs(directory_mesh, exist_ok=True)

def test():
    images = rgbs[np.newaxis, :, :, :]  # (1, 4, height, width)
    images = torch.from_numpy(images.astype('float32') / 255.0)
    images = torch.autograd.Variable(images).cuda()
    voxels = np.load(args.voxel_path)['arr_0']

    batch_iou, vertices, faces = model(images, voxels=voxels, task='test')
    iou = batch_iou

    mesh_path = os.path.oin(directory_mesh, '%06d.obj' % MESH_ID)
    input_path = os.path.join(directory_mesh, '%06d.png' % MESH_ID)
    srf.save_obj(mesh_path, vertices[0], faces[0])
    imageio.imsave(input_path, img_cvt(images[0]))


test()
