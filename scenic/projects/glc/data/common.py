from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from PIL import Image
import tifffile

if TYPE_CHECKING:
    import numpy.typing as npt

    Patches = npt.NDArray

def get_num_channels(data):
    counter=0
    if data=="all":
        counter+=6
    if "rgb" in data:
        counter +=3
    if "near_ir" in data:
        counter+=1
    if "altitude" in data:
        counter += 1
    if "landcover" in data:
        counter += 1
    return(counter)

def load_patch( 
    observation_id: Union[int, str],
    patches_path: Union[str, Path],
    *,
    data: Union[str, list[str]] = "all",
    landcover_mapping: Optional[npt.NDArray] = None,
    return_arrays: bool = True,
) -> list[Patches]:
    """Loads the patch data associated to an observation id

    Parameters
    ----------
    observation_id : integer / string
        Identifier of the observation.
    patches_path : string / pathlib.Path
        Path to the folder containing all the patches.
    data : string or list of string
        Specifies what data to load, possible values: 'all', 'rgb', 'near_ir', 'landcover' or 'altitude'.
    landcover_mapping : 1d array-like
        Facultative mapping of landcover codes, useful to align France and US codes.
    return_arrays : boolean
        If True, returns all the patches as Numpy arrays (no PIL.Image returned).

    Returns
    -------
    patches : list of size 4 containing 2d array-like objects
        Returns a list containing all the patches in the following order: RGB, Near-IR, altitude and landcover.
    """
    observation_id = str(observation_id)

    region_id = observation_id[0]
    if region_id == "1":
        region = "patches-fr"
    elif region_id == "2":
        region = "patches-us"
    else:
        raise ValueError(
            "Incorrect 'observation_id' {}, can not extract region id from it".format(
                observation_id
            )
        )

    subfolder1 = observation_id[-2:]
    subfolder2 = observation_id[-4:-2]

    filename = Path(patches_path) / region / subfolder1 / subfolder2 / observation_id

    patches = []

    if data == "all":
        data = ["rgb", "near_ir", "landcover", "altitude"]

    if "rgb" in data:
        rgb_filename = filename.with_name(filename.stem + "_rgb.jpg")
        rgb_patch = Image.open(rgb_filename)
        if return_arrays:
            rgb_patch = np.asarray(rgb_patch)
        patches.append(rgb_patch)

    if "near_ir" in data:
        near_ir_filename = filename.with_name(filename.stem + "_near_ir.jpg")
        near_ir_patch = Image.open(near_ir_filename)
        if return_arrays:
            near_ir_patch = np.asarray(near_ir_patch)
        patches.append(near_ir_patch)

    if "altitude" in data:
        altitude_filename = filename.with_name(filename.stem + "_altitude.tif")
        altitude_patch = tifffile.imread(altitude_filename)
        patches.append(altitude_patch)

    if "landcover" in data:
        landcover_filename = filename.with_name(filename.stem + "_landcover.tif")
        landcover_patch = tifffile.imread(landcover_filename)
        if landcover_mapping is not None:
            landcover_patch = landcover_mapping[landcover_patch]
        patches.append(landcover_patch)

    return patches
