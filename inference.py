import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import datasets
from utils import flow_viz
from utils import frame_utils

from glob import glob
import cv2

from flowdiffuser import FlowDiffuser

from utils.utils import InputPadder, forward_interpolate

def prepare_image(fn1, fn2, keep_size=True):
    print(f"preparing image...")
    print(f"fn = {fn1}, {fn2}")

    image1 = frame_utils.read_gen(fn1)
    image2 = frame_utils.read_gen(fn2)
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]
    
    dsize = image1.shape[0:2]
    if image1.shape[0]>=1080:   # FlowDiffuser get "CUDA out of memory" for > 1080p
        dsize = (image1.shape[1]//2, image1.shape[0]//2)
        image1 = cv2.resize(image1, dsize=dsize, interpolation=cv2.INTER_LINEAR)
        image2 = cv2.resize(image2, dsize=dsize, interpolation=cv2.INTER_LINEAR)
    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    return image1, image2

def visualize_flow(vis_dir, model, img_pairs, iters=32):
    weights = None
    idx = 0
    for img_pair in img_pairs:
        fn1, fn2 = img_pair
        print(f"processing {fn1}, {fn2}...")

        image1, image2 = prepare_image(fn1, fn2)

        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flow_low, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()



        # Write *.flo file
        #filename = osp.splitext(osp.basename(viz_fn))[0]
        #frame_utils.writeFlow(osp.join(vis_dir, filename + ".flo"), flow)
        flow_file = 'flow_{:04}_to_{:04}'.format(idx + 1, idx + 2)
        frame_utils.writeFlow(os.path.join(vis_dir, flow_file + ".flo"), flow)

        flow_img = flow_viz.flow_to_image(flow)
        cv2.imwrite(os.path.join(vis_dir, flow_file + '.png'), flow_img[:, :, [2,1,0]])
        idx+=1

def process_subfolder(subfolder):
    img_pairs = []
    image_list = sorted(glob(os.path.join(subfolder, '*.png')))
    for i in range(len(image_list)-1):
        img_pairs.append((image_list[i], image_list[i+1]))

    return img_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Added to support common args with the other flow algorithms
    parser.add_argument('--seq_dir', help="folder for input images. If there are subfolders, it will loop through them", default='default')
    parser.add_argument('--vis_dir', help="output folder which will follow the structure of input seq_dir", default='default')
    parser.add_argument('--firstfile', type=str, help='first image in seq_dir (optional)')
    parser.add_argument('--secondfile', type=str, help='second image in seq_dir (optional)')

    args = parser.parse_args()

    model = torch.nn.DataParallel(FlowDiffuser(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # Calculate flow: Iterate over files/subfolders under args.seq_dir
    orig_seq_dir = args.seq_dir
    orig_vis_dir = args.vis_dir
    dirList = sorted(os.scandir(args.seq_dir), key=lambda e: e.name)
    #for entry in os.scandir(args.seq_dir):
    for entry in dirList:
        if entry.name.startswith('.'):
            continue
        # If seq_dir includes other directories (like processed BMS data), iterate each directory
        if entry.is_dir():
            print ("Directory: ", entry.name)
            args.seq_dir = orig_seq_dir + '/' + entry.name
            args.vis_dir = orig_vis_dir + '/' + entry.name

            if not os.path.exists(args.vis_dir):
                os.makedirs(args.vis_dir)

            #img_pairs = process_sintel(args.seq_dir)
            img_pairs = process_subfolder(args.seq_dir)

            with torch.no_grad():
                visualize_flow(args.vis_dir, model, img_pairs)
                #inference(model, args, device=device)
        else:
            print (entry.name)
            img_pairs = process_subfolder(args.seq_dir)

            with torch.no_grad():
                visualize_flow(args.vis_dir, model, img_pairs)
                #inference(model, args, device=device)
            break


