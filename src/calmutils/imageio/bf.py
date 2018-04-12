import javabridge
import bioformats


def read_bf(path):
    """

    read an image into a np-array using BioFormats

    Parameters
    ----------
    path: str
        file path to read

    Returns
    -------
    img: np.array
        image as np-array
    """
    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
    img = bioformats.load_image(path, rescale=False)
    return img
