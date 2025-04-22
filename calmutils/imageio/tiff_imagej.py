from tifffile import imwrite, TiffFile
import numpy as np


IMAGEJ_AXES = 'TZCYX'


def save_tiff_imagej(path, img, axes=None, pixel_size=None, distance_unit=None, existing_metadata=None):

    """
    Convenience function to save TIFF files so basic metadata (pixel size, dimension order, ..) will be available in ImageJ.
    Note that dimensions in the file written may be transposed to conform to the TZXYC order of ImageJ.
    """

    if axes is None:
        axes = IMAGEJ_AXES
    
    # make sure axes is upper case
    axes = axes.upper()

    if pixel_size is None:
        pixel_size = [1] * 3
    if len(pixel_size) < 3:
        pixel_size = [1] * (3-len(pixel_size)) + list(pixel_size)

    if distance_unit is None:
        distance_unit = 'pixels'

    # indices of axes in ImageJ order
    imagej_axes_selection = [IMAGEJ_AXES.index(a) for a in axes[-img.ndim:]]

    # subset of axes labels in ImageJ order
    imagej_axes_subset = ''.join(IMAGEJ_AXES[i] for i in sorted(imagej_axes_selection))

    # reorder indices to transpose image into ImageJ convention
    axis_reorder_for_imagej = np.argsort(imagej_axes_selection)
    img = np.transpose(img, axis_reorder_for_imagej)

    resolution_xy = (1/pixel_size[2], 1/pixel_size[1])
    metadata = {} if existing_metadata is None else existing_metadata
    metadata.update({'spacing': pixel_size[0], 'unit': distance_unit, 'axes': imagej_axes_subset})

    # # FIXME: does not seem to show up in ImageJ
    # # time_interval = 0.1
    # # if time_interval is not None:
    # #     metadata['finterval']: time_interval
    # #     metadata['fps']: 1/time_interval

    imwrite(path, img, imagej=True, resolution=resolution_xy, metadata=metadata)


def get_imagej_tiff_metadata(path):
    """
    Quick helper to get ImageJ metadata dict from a TIFF file.
    Returns None if file is not ImageJ-format.
    """
    with TiffFile(path) as reader:
        if not reader.is_imagej:
            return None
        return reader.imagej_metadata


def get_imagej_tiff_pixel_size(image_path):
    """
    Get pixel size and unit name from ImageJ-format TIFF.
    Following https://forum.image.sc/t/reading-pixel-size-from-image-file-with-python/74798/1
    """
    with TiffFile(image_path) as reader:

        # error on non-ImageJ. TODO: maybe try regardless?
        if not reader.is_imagej:
            raise ValueError("Image does not seem to be ImageJ-format.")

        # get unit and z-spacing from ImageJ metadata
        ij_meta = reader.imagej_metadata
        pixel_unit = ij_meta.get("unit", "pixel")
        pixel_size = [ij_meta.get("spacing", 1.0)]

        # get YX sizes from first plane tags
        for dim in "YX":
            num_pixels, units = reader.pages[0].tags[f'{dim}Resolution'].value
            pixel_size.append(units / num_pixels)

        return np.array(pixel_size), pixel_unit