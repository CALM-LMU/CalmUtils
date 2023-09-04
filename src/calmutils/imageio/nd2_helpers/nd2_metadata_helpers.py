from operator import sub
# TODO: unify to one library (preferentially nd2)
from nd2 import ND2File
from nd2reader import ND2Reader

def get_pixel_size(file_path):
    with ND2File(file_path) as reader:
        # invert xyz voxel size to zyx to match classical numpy array order
        pixel_size = reader.voxel_size()[::-1]
    return pixel_size


def get_z_direction(file_path):
    with ND2Reader(file_path) as reader:
        if not 'z_coordinates' in reader.metadata or len(reader.metadata['z_coordinates']) < 2:
            return None
        # difference of z position of first two planes -> z-spacing
        psz_z = sub(*reader.metadata['z_coordinates'][:2])
        z_direction = 'to_sample' if psz_z > 0 else 'from_sample'
    return z_direction


if __name__ == '__main__':
    f = '/Users/david/Desktop/23AM03-02_4001.nd2'
    print(get_pixel_size(f), get_z_direction(f))
    f = '/Users/david/Desktop/Beads_single005.nd2'
    print(get_pixel_size(f), get_z_direction(f))
    