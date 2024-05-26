__copyright__ = \
    """
    Copyright (C) 2022 University of Liège, Gembloux Agro-Bio Tech, Forest Is Life
    All rights reserved.

    This source code is under the CC BY-NC-SA-4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/). 
    It is to be used for academic research purposes only, no commercial use is permitted.

    Please contact the author Alexandre Delplanque (alexandre.delplanque@uliege.be) for any questions.

    Last modification: March 29, 2023
    """
__author__ = "Alexandre Delplanque"
__license__ = "CC BY-NC-SA 4.0"
__version__ = "0.2.0"


import argparse
import os
import PIL
import torchvision
import numpy
import cv2
from pathlib import Path
import math

from albumentations import PadIfNeeded
from itertools import product
from torchvision.utils import save_image

from tqdm import tqdm

from animaloc.data import ImageToPatches, PatchesBuffer, save_batch_images

parser = argparse.ArgumentParser(prog='patcher', description='Cut images into patches')

parser.add_argument('root', type=str,
    help='path to the images directory (str)')
parser.add_argument('height', type=int,
    help='height of the patches, in pixels (int)'
    )
parser.add_argument('width', type=int,
    help='width of the patches, in pixels (int)'
    )
parser.add_argument('overlap', type=int,
    help='overlap between patches, in pixels (int)')
parser.add_argument('-overlapfactor', type=float,default=0.0,
    help='overlap ratio between patches, in [0,1]. It works only for raw images. It does not work with -csv')
parser.add_argument('-ratioheight',type=float,default=0.0,
                    help='ratios for height. When it is not 1.0 then the height of the tile is infered from the image height and the ratio.')
parser.add_argument('-ratiowidth',type=float,default=0.0,
                    help='ratios for width. When it is not 1.0 then the width of the tile is infered from the image width and the ratio.')
parser.add_argument('-dest', type=str,
    help='destination path (str)')
parser.add_argument('-csv', type=str,
    help='path to a csv file containing annotations (str). Defaults to None')
parser.add_argument('-min', type=float, default=0.1,
    help='minimum fraction of area for an annotation to be kept (float). Defautls to 0.1')
parser.add_argument('-all', type=bool, default=False,
    help='set to True to save all patches, not only those containing annotations (bool). Defaults to False')
parser.add_argument('-pattern',type=str,default='**/*.jpg',
                    help='pattern of files extension')
parser.add_argument('-rmheight',type=float,default=0.0,
                    help='height overlap to be removed at both bottom and top')
parser.add_argument('-rmwidth',type=float,default=0.0,
                    help='width overlap to be removed at both left and right sides')


args = parser.parse_args()

#Helper funcs
def get_patches(image,tile_w:int,tile_h:int,overlaping_factor:float):
    
    patches = list()
    image_width=image.shape[2]
    image_height=image.shape[1]

    def get_coordinates():

        # x limits
        lim = math.ceil((image_width-tile_w)/((1-overlaping_factor)*tile_w))
        x_right = [math.floor(tile_w + i*(1-overlaping_factor)*tile_w) for i in range(lim)]
        x_coords = [(x-tile_w,x) for x in x_right]
        if len(x_coords)>0:
            left,right = x_coords[-1]
            x_coords[-1] = (left,image_width) # extending to remaining pixels

        # y limits
        lim = math.ceil((image_height-tile_h)/((1-overlaping_factor)*tile_h))
        y_bottom = [math.floor(tile_h + i*(1-overlaping_factor)*tile_h) for i in range(lim)]
        y_coords = [(y-tile_h,y) for y in y_bottom]
        if len(y_coords)>0:
            top,bottom = y_coords[-1]
            y_coords[-1] = (top,image_height) # extending to remaining pixels

        # tiles coordinates
        if len(y_coords)>0 and len(x_coords)>0:
            pass
        elif len(y_coords) == 0:
            y_coords = [(0,image_height),]
        elif len(x_coords) == 0:
            x_coords = [(0,image_width),]

        coordinates = product(x_coords,y_coords)
        return list(coordinates)
    
    coords =  get_coordinates()

    # store patches
    for (x_left,x_right),(y_top,y_bottom) in coords:
        patches.append(image[:,y_top:y_bottom,x_left:x_right])
        
    return patches

def save_list_images(
    batch:list,
    basename: str,
    dest_folder: str
    ) -> None:
    ''' Save mini-batch tensors into image files

    Use torchvision save_image function,
    see https://pytorch.org/vision/stable/utils.html#torchvision.utils.save_image

    Args:
        batch (list): mini-batch tensor
        basename (str) : parent image name, with extension
        dest_folder (str): destination folder path
    '''

    base_wo_extension, extension = basename.split('.')[0], basename.split('.')[1]
    for i, b in enumerate(range(len(batch))):
        full_path = '_'.join([base_wo_extension, str(i) + '.']) + extension
        save_path = os.path.join(dest_folder, full_path)
        save_image(batch[b], fp=save_path)


def main():

    # images_paths =list(Path(args.root).glob(args.pattern))  #[os.path.join(args.root, p) for p in os.listdir(args.root) if not p.endswith('.csv')]
    images_paths = [p for p in Path(args.root).glob(args.pattern)]

    if args.csv is not None:
        patches_buffer = PatchesBuffer(args.csv, args.root, (args.height, args.width), overlap=args.overlap, min_visibility=args.min).buffer
        patches_buffer.drop(columns='limits').to_csv(os.path.join(args.dest, 'gt.csv'), index=False)

    for img_path in tqdm(images_paths, desc='Exporting patches'):
        try:
            pil_img = PIL.Image.open(img_path)
        except :
            print("failed for: ",img_path,flush=True)
            continue
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)


        # Cropping out image-level overlap
        height_overlap = math.ceil(args.rmheight * img_tensor.shape[1])
        width_overlap = math.ceil(args.rmwidth * img_tensor.shape[2])
        if height_overlap*width_overlap > 0 :
            img_tensor = img_tensor[:,height_overlap:-height_overlap, width_overlap:-width_overlap]
            print(f"Removing {2*width_overlap} pixels to the width; and {2*height_overlap} pixels to the height.")
        elif (height_overlap == 0) and (width_overlap != 0):
            img_tensor = img_tensor[:,:, width_overlap:-width_overlap]
        elif width_overlap == 0 and (height_overlap != 0):
            img_tensor = img_tensor[:,height_overlap:-height_overlap,:]
        
        # Computes tile width and height using the given ratios
        # It overrides the parameters 'width' and 'height'
        assert (args.ratiowidth<=1.0) and (args.ratioheight<=1.0), "The ratios should be at most 1.0"
        if args.ratiowidth > 0.0:
            args.width = math.ceil(img_tensor.shape[2]*args.ratiowidth)
        if args.ratioheight > 0.0:
            args.height = math.ceil(img_tensor.shape[1]*args.ratioheight)
        # checking overlapfactor provided
        assert args.overlapfactor<1, 'It should be less than 1.'

        if args.csv is not None:
            # save all patches
            if args.all:
                if args.overlapfactor>0 :
                    patches = get_patches(img_tensor,tile_w=args.width,tile_h=args.height,overlaping_factor=args.overlapfactor)
                    save_batch_images(patches, img_name, args.dest)
                else:
                    patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
                    save_list_images(patches, img_name, args.dest)

            # or only annotated ones
            else:
                padder = PadIfNeeded(
                    args.height, args.width,
                    position = PadIfNeeded.PositionType.TOP_LEFT,
                    border_mode = cv2.BORDER_CONSTANT,
                    value= 0
                    )
                img_ptch_df = patches_buffer[patches_buffer['base_images']==img_name]
                for row in img_ptch_df[['images','limits']].to_numpy().tolist():
                    ptch_name, limits = row[0], row[1]
                    cropped_img = numpy.array(pil_img.crop(limits.get_tuple))
                    padded_img = PIL.Image.fromarray(padder(image = cropped_img)['image'])
                    padded_img.save(os.path.join(args.dest, ptch_name))

        else:
            if args.overlapfactor > 0:
                patches = get_patches(img_tensor,tile_w=args.width,tile_h=args.height,overlaping_factor=args.overlapfactor)
                save_list_images(patches, img_name, args.dest)
            else:
                patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
                save_batch_images(patches, img_name, args.dest)


if __name__ == '__main__':
    main()