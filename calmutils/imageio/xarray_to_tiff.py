from pathlib import Path

from .tiff_imagej import save_tiff_imagej


def xarray_to_tiffs(
    img,
    out_directory,
    out_prefix,
    split_dimensions=("C", "T", "P"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1,
    pixel_size = None,
):

    # if no pixel sizes are given, guess from xarray coordinate ticks (difference between adjacent)
    if pixel_size is None:
        psz_x = (img.coords['X'].values[1:] - img.coords['X'].values[:-1]).mean() if 'X' in img.coords else 1
        psz_y = (img.coords['Y'].values[1:] - img.coords['Y'].values[:-1]).mean() if 'Y' in img.coords else 1
        psz_z = (img.coords['Z'].values[1:] - img.coords['Z'].values[:-1]).mean() if 'Z' in img.coords else 1
        pixel_size = [psz_z, psz_y, psz_x]

    # find which of the selected split dimensions are present
    present_split_dimensions = [d for d in split_dimensions if d in img.dims]

    # handle no splitting
    if len(present_split_dimensions) == 0:

        # construct filename, dimension names
        out_filename = out_prefix + '.tif'
        out_filename = Path(out_directory) / out_filename
        axes = "".join([d for d in img.dims])

        save_tiff_imagej(
            out_filename,
            img.values.squeeze(),
            axes=axes,
            distance_unit="micron",
            pixel_size=pixel_size
        )

        return

    # group by split dimensions
    for idx, sub_img in img.groupby(present_split_dimensions):

        # treat even single split dimension as list of one
        if len(present_split_dimensions) == 1:
            idx = [idx]

        # get integer indices if desired
        if use_indices:
            filename_idx = [img.get_index(d).get_loc(i) for d,i in zip(present_split_dimensions, idx)]
            filename_idx = [str(i).rjust(min_index_len, '0') for i in filename_idx]
        else:
            filename_idx = idx

        # construct out filename
        out_filename = out_prefix + "".join(prefixes[d] + i for d, i in zip(present_split_dimensions, filename_idx)) + '.tif'
        out_filename = Path(out_directory) / out_filename

        # save as tiff
        axes = "".join([d for d in img.dims if d not in split_dimensions])
        save_tiff_imagej(out_filename, sub_img.values.squeeze(), axes=axes, distance_unit="micron", pixel_size=pixel_size)