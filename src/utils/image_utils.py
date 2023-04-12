import png
import numpy as np
from PIL import Image

def load_png_image(path: str) -> np.ndarray:
    """
    Import a PNG image as a numpy ndarray.

    :param path: The path to the PNG image file.
    :return: A numpy ndarray containing the image data. The dimensions of the array will be (height, width, channels),
    where channels is either 3 (for RGB images) or 4 (for RGBA images).
    """
    # Open the file in binary mode
    with open(path, 'rb') as f:
        # Use the png library's reader to read the PNG file
        reader = png.Reader(file=f)
        # The reader.read() function returns a tuple containing the width, height, pixel data, and metadata
        w, h, pixels, metadata = reader.read()
        # Check if the image has an alpha channel (transparency)
        if metadata['alpha']:
            channels = 4  # RGBA image
        else:
            channels = 3  # RGB image
        # Convert the pixel data from a generator to a 2D numpy array
        image_2d = np.vstack(map(np.uint16, pixels))
        # Reshape the 2D numpy array into a 3D numpy array with shape (height, width, channels)
        image = np.reshape(image_2d, (h, w, channels))

        return image


def load_jpg_image(path: str) -> np.ndarray:
    """
    Import a JPG image as a numpy ndarray.

    :param path: The path to the JPG image file.
    :return: A numpy ndarray containing the image data. The dimensions of the array will be (height, width, channels),
    where channels is 3 for RGB images.
    """
    # Use the Image library from PIL to open the JPG image file
    with Image.open(path) as image:
        # Convert the image to a numpy ndarray
        image_np = np.array(image)

        return image_np