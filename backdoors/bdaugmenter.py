"""
file: bdaugmenter.py
Purpose: The file defines the augmenter used for the preprocessing step used
         for inference.
"""
import imgaug as ia
from imgaug import augmenters as iaa

augmenter = iaa.Sequential(
    [
	iaa.Fliplr(0.5), # horizontally flip 50% of all images

        iaa.Sometimes(0.5, iaa.Affine(
            rotate=(-10, 10), # randomly rotate some of the image
            mode=ia.ALL
        ))
    ],
)
