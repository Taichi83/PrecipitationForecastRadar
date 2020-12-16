import os
import datetime
import pytz
from typing import List, Optional, Tuple, Dict
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

import rasterio
from rasterio.windows import Window, transform
import pandas as pd

from PrecipitationForecastRadar.dataset.utils import get_all_file_path_s3, copy_to_s3, check_file_existence_local, \
    argwrapper, \
    imap_unordered_bar


def get_array(path_img: str, lon: float, lat: float, size: Tuple[int, int] = (256, 256), pos: str = 'center',
              band: Optional[int] = 1) -> Tuple[np.array, rasterio.profiles.Profile, Window]:
    """ Get specific sized square array from geotiff data

    Args:
        path_img (str): image path or url
        lon (float): longitude
        lat (float): latitude
        size (Tuple[int, int]): numpy array size
        pos (str): If 'center', lon and lat are center position in acquired square
        band (Optional[int]): If None, all band.

    Returns:
        (Tuple[np.array, rasterio.profiles.Profile, Window]): numpy array, rasterio profile, rasterio, window

    """
    src = rasterio.open(path_img)
    py, px = src.index(lon, lat)
    if pos == 'center':
        px = px - size[0] // 2
        py = py - size[1] // 2
    window = Window(px, py, size[1], size[0])
    out_profile = src.profile.copy()
    if band is not None:
        clip = src.read(band, window=window)
    else:
        clip = src.read(window=window)

    return clip, out_profile, window


def get_cropped_gtiff(path_img: str, path_out: str, lon: float, lat: float, array_size: Tuple[int, int] = (256, 256),
                      pos: str = 'center', band: Optional[int] = 1) -> Tuple[float, float]:
    """

    Args:
        path_img (str): image path or url
        path_out (str): saved path on local
        lon (float): longitude
        lat (float): latitude
        array_size (Tuple[int, int]): numpy array size
        pos (str): If 'center', lon and lat are center position in acquired square
        band (Optional[int]): If None, all band.

    Returns:

    """
    # test
    src = rasterio.open(path_img)

    py, px = src.index(lon, lat)
    if pos == 'center':
        px = px - array_size[0] // 2
        py = py - array_size[1] // 2
    window = Window(px, py, array_size[1], array_size[0])
    out_profile = src.profile.copy()

    # temp
    transform_window = transform(window, src.transform)
    out_profile["transform"] = transform_window

    if band is not None:
        clip = src.read(band, window=window)
        out_profile.update(count=1,
                           height=clip.shape[0],
                           width=clip.shape[1])

        with rasterio.open(path_out, 'w', **out_profile) as dst:
            dst.write(clip, 1)
    else:
        clip = src.read(window=window)
        out_profile.update(height=clip.shape[1],
                           width=clip.shape[2])
        with rasterio.open(path_out, 'w', **out_profile) as dst:
            dst.write(clip)

    return np.sum(clip == src.nodata) / clip.size, np.sum(clip > 0) / clip.size


def get_datetime(path_img: str) -> datetime.datetime:
    """ Get datetime of given file

    Args:
        path_img (str): target file path

    Returns:
        (datetime.datetime): Acquired date of the target file

    """

    filename = os.path.basename(path_img)
    datetime_str = filename.split('_')[4]
    return datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S%f').replace(tzinfo=pytz.utc)


def check_filename_in_time_range(path_img: str, datetime_start: datetime.datetime,
                                 datetime_end: datetime.datetime) -> bool:
    """ Check the given file's acquired datetime is between time range

    Args:
        path_img (str): target file path
        datetime_start (datetime.datetime): start time of time range
        datetime_end (datetime.datetime): start time of time range

    Returns:
        (bool): If True, the acquired time is in time range

    """
    datetime_target = get_datetime(path_img)
    return (datetime_start <= datetime_target) and (datetime_target <= datetime_end)


class DatasetMaker(object):
    def __init__(self, dir_parent_src: str, dir_parent_dst_local: str, dir_parent_dst_s3: str, subdir_dst: str,
                 src_is_s3: bool = True):
        """ Make dataset for PyTorch pipeline from multiple files on S3

        Args:
            dir_parent_src (str): local or http of parent directory of target files
            dir_parent_dst_local (str): destination directory for output files on local
            dir_parent_dst_s3 (str): destination directory for output files on S3
            subdir_dst (str): name of sub directory under the parent directory
            src_is_s3 (bool): If True, dir_parent_src is in s3
        """

        self.dir_parent_src = dir_parent_src
        self.dir_parent_dst_local = dir_parent_dst_local
        self.dir_parent_dst_s3 = dir_parent_dst_s3
        self.subdir_dst = subdir_dst
        self.src_is_s3 = src_is_s3

        self.ext_filter = '_grib2_reproj-4326.tif'
        self.cropped_name = 'cropped'
        self.list_dict_local = []
        self.list_dict_s3 = []

        self.key_local = 'path_image'
        self.key_s3 = 'url_s3'
        self.col_path = 'path_image'

        self.file_list_name = 'file_list.csv'
        self.parameter_list = 'parameters.json'

    def _get_candidate_files_path(self, datetime_start: datetime.datetime, datetime_end: datetime.datetime) -> List[
        str]:
        """ filter files under self.dir_parent_src with self.ext_filter and datetime range

        Args:
            datetime_start (datetime.datetime): start datetime for filtering
            datetime_end (datetime.datetime): end datetime for filtering

        Returns:
            (List[str]): filtered file path list
        """
        if self.src_is_s3:
            kwargs_filter = {
                'datetime_start': datetime_start,
                'datetime_end': datetime_end,
            }
            files_path = get_all_file_path_s3(dir_parent=self.dir_parent_src,
                                              ext_filter=self.ext_filter,
                                              func_kwargs=(check_filename_in_time_range, kwargs_filter)
                                              )
        else:
            p = Path(self.dir_parent_src)
            files_path = []
            for path_temp in p.glob("**/*" + self.ext_filter):
                if check_filename_in_time_range(path_temp, datetime_start, datetime_end):
                    files_path.append(path_temp)

        return files_path

    def get_cropped_tiff_upload(self, path_img: str, lon: float, lat: float,
                                array_size: Tuple[int, int] = (256, 256), pos: str = 'center', band: Optional[int] = 1,
                                overwrite: bool = True, s3_upload: bool = True, remove_local_file: bool = False,
                                multiprocessing: bool = False) -> Dict:
        """ Get cropped geotiff image from geotiff image

        Args:
            path_img (str): local path or url of target geotiff file
            lon (float): longitude
            lat (float): latitude
            array_size (Tuple[int, int]): numpy array size
            pos (str): If 'center', lon and lat are center position in acquired square
            band (Optional[int]): If None, all band.
            overwrite (bool): If True, avoid downloading the file with the same name and not empty
            s3_upload (bool): If True, the local file is uploaded to S3
            remove_local_file (bool): If True, the local file will be removed after uploading to S3
            multiprocessing (bool): If True, this def can be in the part of multiprocessing

        Returns:
            (Dict): file path info

        """

        filename = os.path.basename(path_img)
        filename = os.path.splitext(filename)[0] + '_' + self.cropped_name + os.path.splitext(filename)[1]

        path_out = os.path.join(self.dir_parent_dst_local, self.subdir_dst, filename)

        if overwrite or not check_file_existence_local(path_out):
            rate_nodata, rate_rain = get_cropped_gtiff(path_img=path_img,
                                                       path_out=path_out,
                                                       lon=lon, lat=lat, array_size=array_size, pos=pos, band=band)
        else:
            src = rasterio.open(path_out)
            array = src.read(1)
            rate_nodata = np.sum(array == src.nodata) / array.size
            rate_rain = np.sum(array > 0) / array.size

        if s3_upload:
            path_dst_s3 = os.path.join(self.dir_parent_dst_s3, self.subdir_dst, filename)
            url_out = copy_to_s3(path_src_local=path_out, path_dst_s3=path_dst_s3, remove_local_file=remove_local_file,
                                 overwrite=overwrite, multiprocessing=multiprocessing)
            if remove_local_file:
                path_out = None
        else:
            url_out = None

        dict_path_info = {
            self.key_local: path_out,
            'url_s3_origin': path_img,
            self.key_s3: url_out,
            'datetime': get_datetime(path_img),
            'file_name': filename,
            'rate_nodata': rate_nodata,
            'rate_rain': rate_rain
        }
        return dict_path_info

    def _get_list_of_files(self, files_path_database, files_path_origin, support_info):
        if len(files_path_database) == 0:
            return []

        assert len(files_path_database) != len(files_path_origin), print('check length of input args')

        list_info = []
        for i, file_path_database in enumerate(files_path_database):
            file_path_origin = files_path_origin[i]
            datetime_target = get_datetime(file_path_origin)
            dict_info = {
                'path_image': files_path_database,
                'path_origin': files_path_origin,
                'datetime': datetime_target,
            }
            dict_info.update(support_info)
            list_info.append(dict_info)
        return list_info

    def _get_organized_df(self, list_dict):
        df_temp = pd.DataFrame(list_dict)
        col_base = list(df_temp.columns)
        col_base.remove(self.key_local)
        col_base.remove(self.key_s3)
        df_base = df_temp.loc[:, col_base]
        return df_base

    def prepare_dataset(self, datetime_start: datetime.datetime, datetime_end: datetime.datetime,
                        lon: float, lat: float,
                        array_size: Tuple[int, int] = (256, 256), pos: str = 'center', band: Optional[int] = 1,
                        overwrite: bool = True,
                        s3_upload: bool = True, remove_local_file: bool = False, processes: int = 1
                        ) -> Tuple[pd.DataFrame, Dict]:
        """ Prepare dataset with cropping images

        Args:
            datetime_start (datetime.datetime): start time of time range
            datetime_end (datetime.datetime): end time of time range
            lon (float): longitude
            lat (float): latitude
            array_size (Tuple[int, int]): numpy array size
            pos (str): If 'center', lon and lat are center position in acquired square
            band (Optional[int]): If None, all band.
            overwrite (bool): If True, avoid downloading the file with the same name and not empty
            s3_upload (bool): If True, the local file is uploaded to S3
            remove_local_file (bool): If True, the local file will be removed after uploading to S3
            processes (int): the number of threads for multiprocessing

        Returns:
            (Tuple[pd.DataFrame, Dict]): file info in df, and parameters in json

        """

        dict_parameters = {
            'longitude': lon,
            'latitude': lat,
            'position': pos,
            'array_size': array_size,
            'band': band
        }

        files_path = self._get_candidate_files_path(datetime_start=datetime_start, datetime_end=datetime_end)
        print(files_path)

        dir_out = os.path.join(self.dir_parent_dst_local, self.subdir_dst)
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)

        if processes == 1:
            list_dict_path_info = []
            for file_path in tqdm(files_path, total=len(files_path)):
                dict_path_info = self.get_cropped_tiff_upload(path_img=file_path,
                                                              lon=lon, lat=lat,
                                                              array_size=array_size,
                                                              pos=pos,
                                                              band=band,
                                                              overwrite=overwrite,
                                                              s3_upload=s3_upload,
                                                              remove_local_file=remove_local_file,
                                                              multiprocessing=False)

                list_dict_path_info.append(dict_path_info)

        else:
            func_args = [(self.get_cropped_tiff_upload, files_path, lon, lat, array_size, pos, band, overwrite,
                          s3_upload, remove_local_file, True) for files_path in files_path]
            list_dict_path_info = imap_unordered_bar(argwrapper, func_args, processes, extend=False)

        df_file_list = self._get_organized_df(list_dict_path_info)

        # save dataframe
        path_out = os.path.join(self.dir_parent_dst_local, self.subdir_dst, self.file_list_name)
        df_file_list.to_csv(path_out, index=False)
        if s3_upload:
            path_out_s3 = os.path.join(self.dir_parent_dst_s3, self.subdir_dst, self.file_list_name)
            copy_to_s3(path_src_local=path_out, path_dst_s3=path_out_s3, remove_local_file=remove_local_file,
                       overwrite=overwrite, multiprocessing=False)

        # save parameter
        path_out = os.path.join(self.dir_parent_dst_local, self.subdir_dst, self.parameter_list)
        with open(path_out, mode='w') as file:
            json.dump(dict_parameters, file)

        if s3_upload:
            path_out_s3 = os.path.join(self.dir_parent_dst_s3, self.subdir_dst, self.parameter_list)
            copy_to_s3(path_src_local=path_out, path_dst_s3=path_out_s3, remove_local_file=remove_local_file,
                       overwrite=overwrite, multiprocessing=False)

        return df_file_list, dict_parameters


if __name__ == '__main__':
    dir_parent_src = 'data/JMA/RA/converted/RA2015'
    lon = 127.8
    lat = 26.3
    size = (256, 256)

    datetime_start = datetime.datetime(year=2015, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc)
    datetime_end = datetime.datetime(year=2015, month=3, day=31, hour=23, minute=59, tzinfo=pytz.utc)
    dir_parent_dst_local = 'dataset'
    dir_parent_dst_s3 = 'check_data/RA_dataset'
    src_s3 = True
    overwrite = True
    s3_upload = True
    remove_local_file = False
    processes = 10

    dataset_maker = DatasetMaker(dir_parent_src=dir_parent_src, dir_parent_dst_local=dir_parent_dst_local,
                                 dir_parent_dst_s3=dir_parent_dst_s3, subdir_dst='temp', src_is_s3=src_s3)
    df_file_list, dict_parameters = dataset_maker.prepare_dataset(datetime_start=datetime_start,
                                                                  datetime_end=datetime_end,
                                                                  lon=lon, lat=lat,
                                                                  array_size=size,
                                                                  pos='center',
                                                                  band=1,
                                                                  overwrite=overwrite,
                                                                  s3_upload=s3_upload,
                                                                  remove_local_file=remove_local_file,
                                                                  processes=processes)
    print(df_file_list, dict_parameters)
