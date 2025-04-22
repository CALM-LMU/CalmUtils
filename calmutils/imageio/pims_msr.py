import warnings

import numpy as np
import pims

# override download_jar_ default parameters to download last release of loci_tools
pims.bioformats.download_jar.__defaults__ = ('6.7', )

def read_msr_bioformats(file, series_to_read=None):

    data = {}

    # ignore UserWarning about slow jpype version
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        
        # read .msr file with Bioformats
        with pims.Bioformats(file) as reader:
            
            # if no series indices are given, load all
            if series_to_read is None:
                series_to_read = range(reader.size_series)

            for i in series_to_read:
                
                # set series
                reader.series = i

                # gives the series name
                series_name = reader.rdr.getMetadataStore().getImageName(i)

                # pixel sizes in meters
                pixel_sizes = [reader.metadata.PixelsPhysicalSizeZ(i),
                            reader.metadata.PixelsPhysicalSizeY(i),
                            reader.metadata.PixelsPhysicalSizeX(i)]

                # actually load image, squeeze to remove singleton time dimension
                img = np.array(reader).squeeze()

                # build name -> (img, psz) dict
                data[series_name] = (img, pixel_sizes)

    return data