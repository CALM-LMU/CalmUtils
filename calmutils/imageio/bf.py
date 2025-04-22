import warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import functools
from itertools import product

try:
    import javabridge
    import bioformats


    def get_bf_dimensions_single_series(path, series):
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        image_reader = bioformats.get_image_reader(path, path)
        image_reader.rdr.setSeries(series)
        c = image_reader.rdr.getSizeC()
        t = image_reader.rdr.getSizeT()
        d = image_reader.rdr.getSizeZ()
        h = image_reader.rdr.getSizeY()
        w = image_reader.rdr.getSizeX()
        return np.array([c, t, d, h, w])

    def get_bf_num_series(path):
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        image_reader = bioformats.get_image_reader(path, path)
        n_series = image_reader.rdr.getSeriesCount()
        return n_series

    def get_bf_dimensions(path):
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        image_reader = bioformats.get_image_reader(path, path)
        n_series = image_reader.rdr.getSeriesCount()
        
        ppe = ProcessPoolExecutor()
        res =  [f.result() for f in 
                [ppe.submit(get_bf_dimensions_single_series, path, series) for series in range(n_series)]]
        ppe.shutdown()
        return res

    def load_bf_single_plane(path, c, z, t, n, window=None):
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        image_reader = bioformats.get_image_reader(path, path)
        res = image_reader.read(c=c, z=z, t=t, series=n, rescale=False, XYWH=window)
        return res

    def load_series_from_bf_dataset(path, series=0, channels=None, planes=None, timepoints=None, window=None):
        c, t, d, h, w = get_bf_dimensions_single_series(path, series)

        if channels is None:
            channels = (0, c)
        if isinstance(channels, tuple):
            ch_min, ch_max = channels
        else:
            ch_min, ch_max = channels, channels+1
        if ch_max > c:
            raise ValueError('trying to read nonexistant channels.')

        if planes is None:
            planes = (0,d)
        if isinstance(planes, tuple):
            z_min, z_max = planes
        else:
            z_min, z_max = planes, planes+1
        if z_max > d:
            raise ValueError('trying to read nonexistant planes.') 

        if timepoints is None:
            timepoints = (0, t)
        if isinstance(timepoints, tuple):
            t_min, t_max = timepoints
        else:
            t_min, t_max = timepoints, timepoints+1
        if t_max > t:
            raise ValueError('trying to read nonexistant timepoints.')

        if window is not None:
            x,y,w_,h_ = window
            if x+w_ > w or y + h_ > h:
                raise ValueError('trying to read nonexistant window.')

        res = None
        ppe = ProcessPoolExecutor()

        futures = []
        for (c_, t_, z_) in product(range(ch_min, ch_max), range(t_min, t_max), range(z_min, z_max)):
            futures.append(ppe.submit(load_bf_single_plane, path, c_, z_, t_, series, window))

        for f, (c_, t_, z_) in zip(futures, product(range(ch_min, ch_max), range(t_min, t_max), range(z_min, z_max))):
            plane = f.result()
            if res is None:
                res = np.zeros((ch_max-ch_min, t_max-t_min, z_max-z_min, h if window is None else h_, w if window is None else w_), dtype=plane.dtype)
            res[(c_-ch_min, t_-t_min, z_-z_min)] = plane

        ppe.shutdown()
        return res

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

        warnings.warn("calmutils.imageio.read_bf() signature is deprecated and will be renamed.", DeprecationWarning)

        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)
        img = bioformats.load_image(path, rescale=False)
        return img

except ImportError as e:
    print('WARNING: Bioformats bridge not installed, reader functions not available')