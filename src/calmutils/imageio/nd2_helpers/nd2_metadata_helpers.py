from nd2 import ND2File
from nd2.structures import ZStackLoop


def get_pixel_size(file_path):
    with ND2File(file_path) as reader:
        # invert xyz voxel size to zyx to match classical numpy array order
        pixel_size = reader.voxel_size()[::-1]
    return pixel_size

def nd2_get_channel_names_cleaned(file_path, whitespace_replace_char='-'):
    """
    return list of channel (OC) names in ND2 file.
    leading/trailing whitespaces will be stripped and internal whitespaces replaced (with '-' by default)
    """
    with ND2File(file_path) as reader:
        return [s.channel.name.strip().replace(' ', whitespace_replace_char) for s in reader.metadata.channels]

## OLD VERSION with nd2reader
# from operator import sub
# from nd2reader import ND2Reader

# def get_z_direction(file_path):
#     with ND2Reader(file_path) as reader:
#         if not 'z_coordinates' in reader.metadata or len(reader.metadata['z_coordinates']) < 2:
#             return None
#         # difference of z position of first two planes -> z-spacing
#         psz_z = sub(*reader.metadata['z_coordinates'][:2])
#         # NOTE: earlier versions wrongly had to_sample if psz_z > 0, i.e. decreasing positions
#         z_direction = 'to_sample' if psz_z < 0 else 'from_sample'
#     return z_direction

def get_z_direction(nd2_file):
    with ND2File(nd2_file) as fd:
        # try to find z loop metadata, return None if no info can be found
        z_loop_meta = next((loop for loop in fd.experiment if isinstance(loop, ZStackLoop)), None)
        if z_loop_meta is None:
            return None
        # string representation of direction
        return 'bottom_to_top' if z_loop_meta.parameters.bottomToTop else 'top_to_bottom'

if __name__ == '__main__':
    f = '/Users/david/Desktop/23AM03-02_4001.nd2'
    print(get_pixel_size(f), get_z_direction(f))
    f = '/Users/david/Desktop/Beads_single005.nd2'
    print(get_pixel_size(f), get_z_direction(f), nd2_get_channel_names_cleaned(f))
    