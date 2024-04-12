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
parser.add_argument('-ratioheight',type=float,default=1.0,
                    help='ratios for height. When it is not 1.0 then the height of the tile is infered from the image height and the ratio.')
parser.add_argument('-ratiowidth',type=float,default=1.0,
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

args = parser.parse_args()

def main():

    images_paths =list(Path(args.root).glob(args.pattern))  #[os.path.join(args.root, p) for p in os.listdir(args.root) if not p.endswith('.csv')]

    if args.csv is not None:
        patches_buffer = PatchesBuffer(args.csv, args.root, (args.height, args.width), overlap=args.overlap, min_visibility=args.min).buffer
        patches_buffer.drop(columns='limits').to_csv(os.path.join(args.dest, 'gt.csv'), index=False)

    for img_path in tqdm(images_paths, desc='Exporting patches'):
        pil_img = PIL.Image.open(img_path)
        img_tensor = torchvision.transforms.ToTensor()(pil_img)
        img_name = os.path.basename(img_path)

        assert (args.ratiowidth<=1.0) and (args.ratioheight<=1.0), "The ratios should be at most 1.0"

        # computes tile width and height using the given ratios
        # IT overrides the parameters 'width' and 'height'
        if args.ratiowidth < 1.0:
            args.width = math.ceil(img_tensor.shape[2]*args.ratiowidth)
        if args.ratioheight < 1.0:
            args.height = math.ceil(img_tensor.shape[1]*args.ratioheight)


        if args.csv is not None:
            # save all patches
            if args.all:
                patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
                save_batch_images(patches, img_name, args.dest)

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
            patches = ImageToPatches(img_tensor, (args.height, args.width), overlap=args.overlap).make_patches()
            save_batch_images(patches, img_name, args.dest)


if __name__ == '__main__':
    main()