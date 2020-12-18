import datetime
import os
import pytz
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import Dataset


class PrecipitationJMADataset(Dataset):
    def __init__(self,
                 path_img_list: str,
                 num_input_images: int,
                 num_output_images: int,
                 datetime_train_start: Optional[datetime.datetime] = None,
                 datetime_train_end: Optional[datetime.datetime] = None,
                 datetime_test_start: Optional[datetime.datetime] = None,
                 datetime_test_end: Optional[datetime.datetime] = None,
                 train: bool = True,
                 transform=None,
                 precipitation_threshold_train: Optional[float] = None,
                 precipitation_threshold_test: Optional[float] = None,
                 ):

        # super(PrecipitationMap, self).__init__()
        self.path_img_list = path_img_list
        self.num_input_images = num_input_images
        self.num_output_images = num_output_images
        self.sequence_length = num_input_images + num_output_images
        self.datetime_train_start = datetime_train_start
        self.datetime_train_end = datetime_train_end
        self.datetime_test_start = datetime_test_start
        self.datetime_test_end = datetime_test_end
        self.train = train
        self.transform = transform
        self.precipitation_threshold_train = precipitation_threshold_train
        self.precipitation_threshold_test = precipitation_threshold_test

        col_datetime = 'datetime'
        col_filename = 'file_name'
        col_rate_rain = 'rate_rain'

        # get file list in dataframe
        df = pd.read_csv(self.path_img_list)
        df[col_datetime] = pd.to_datetime(df[col_datetime])

        datetime_start = df[col_datetime].min()
        datetime_end = df[col_datetime].max()

        if self.datetime_train_start is None:
            self.datetime_train_start = datetime_start
        if self.datetime_train_end is None:
            self.datetime_train_end = datetime_end
        if self.datetime_test_start is None:
            self.datetime_test_start = datetime_start
        if self.datetime_test_end is None:
            self.datetime_test_end = datetime_end

        df_train = df.loc[
                   (df[col_datetime] <= self.datetime_train_end) & (df[col_datetime] >= self.datetime_train_start), :]
        df_train = df_train.sort_values(by=col_datetime)

        df_test = df.loc[(df[col_datetime] <= self.datetime_test_end) & (df[col_datetime] >= self.datetime_test_start),
                  :]
        df_test = df_test.sort_values(by=col_datetime)

        dir_target = os.path.dirname(self.path_img_list)

        # train
        self.list_img_train = [os.path.join(dir_target, filename) for filename in df_train[col_filename].tolist()]
        self.list_datetime_train = df_train[col_datetime].tolist()
        self.list_rate_rain_train = df_train[col_rate_rain].tolist()
        self.size_dataset_train = len(self.list_img_train) - (num_input_images + num_output_images)

        self.list_path_img_train = []
        for index in range(self.size_dataset_train):
            list_path_img_temp = self.list_img_train[index:index + self.sequence_length]
            if self.precipitation_threshold_train is not None:
                list_rate_rain_temp = self.list_rate_rain_train[index:index + self.sequence_length]
                if sum([rate_rain_temp < self.precipitation_threshold_train for rate_rain_temp in list_rate_rain_temp]) > 0:
                    continue
            self.list_path_img_train.append(list_path_img_temp)

        # test
        self.list_img_test = [os.path.join(dir_target, filename) for filename in df_test[col_filename].tolist()]
        self.list_datetime_test = df_test[col_datetime].tolist()
        self.list_rate_rain_test = df_test[col_rate_rain].tolist()
        self.size_dataset_test = len(self.list_img_test) - (num_input_images + num_output_images)

        self.list_path_img_test = []
        for index in range(self.size_dataset_test):
            list_path_img_temp = self.list_img_test[index:index + self.sequence_length]
            if self.precipitation_threshold_test is not None:
                list_rate_rain_temp = self.list_rate_rain_test[index:index + self.sequence_length]
                if sum([rate_rain_temp < self.precipitation_threshold_test for rate_rain_temp in list_rate_rain_temp]) > 0:
                    continue
            self.list_path_img_test.append(list_path_img_temp)

    def __getitem__(self, index):
        if self.train:
            # list_path_img = self.list_img_train[index:index + self.sequence_length]
            list_path_img = self.list_path_img_train[index]
        else:
            # list_path_img = self.list_img_test[index:index + self.sequence_length]
            list_path_img = self.list_path_img_test[index]

        list_array = []
        for path_img in list_path_img:
            src = rasterio.open(path_img)
            list_array.append(src.read(1))
        imgs = np.array(list_array)

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)

        img_input = imgs[:self.num_input_images]
        img_target = imgs[self.num_input_images:]

        return img_input, img_target

    def __len__(self):
        if self.train:
            # return self.size_dataset_train
            return len(self.list_path_img_train)
        else:
            # return self.size_dataset_test
            return len(self.list_path_img_test)


class PrecipitationJMADataset_org(Dataset):
    def __init__(self,
                 path_img_list: str,
                 num_input_images: int,
                 num_output_images: int,
                 # todo: add datetiem_start_train, _test
                 datetime_start=None,
                 datetime_end=None,
                 # train: bool = True,
                 transform=None):

        # super(PrecipitationMap, self).__init__()
        self.path_img_list = path_img_list
        self.num_input_images = num_input_images
        self.num_output_images = num_output_images
        self.sequence_length = num_input_images + num_output_images
        self.datetime_start = datetime_start
        self.datetime_end = datetime_end
        # self.train = train
        self.transform = transform

        col_datetime = 'datetime'
        col_filename = 'file_name'

        # get file list in dataframe
        df = pd.read_csv(self.path_img_list)
        df[col_datetime] = pd.to_datetime(df[col_datetime])
        df = df.loc[(df[col_datetime] <= self.datetime_end) & (df[col_datetime] >= self.datetime_start), :]
        df = df.sort_values(by=col_datetime)

        dir_target = os.path.dirname(self.path_img_list)
        self.list_img = [os.path.join(dir_target, filename) for filename in df[col_filename].tolist()]
        self.list_datetime = df[col_datetime].tolist()

        self.size_dataset = len(self.list_img) - (num_input_images + num_output_images)

    def __getitem__(self, index):
        list_path_img = self.list_img[index:index + self.sequence_length]
        list_array = []
        for path_img in list_path_img:
            src = rasterio.open(path_img)
            list_array.append(src.read(1))
        imgs = np.array(list_array)

        # add transforms
        if self.transform is not None:
            imgs = self.transform(imgs)

        img_input = imgs[:self.num_input_images]
        img_target = imgs[self.num_input_images:]

        return img_input, img_target

    def __len__(self):
        return self.size_dataset


if __name__ == '__main__':
    path_image_list = 'dataset/jma_okinawa_2015-2016/file_list.csv'
    num_input_images = 6
    num_output_images = 1
    datetime_train_start = datetime.datetime(year=2015, month=1, day=1, tzinfo=pytz.utc)
    datetime_train_end = datetime.datetime(year=2015, month=3, day=31, tzinfo=pytz.utc)
    datetime_test_start = datetime.datetime(year=2016, month=1, day=1, tzinfo=pytz.utc)
    datetime_test_end = datetime.datetime(year=2016, month=3, day=31, tzinfo=pytz.utc)

    precipitation_jma_dataset = PrecipitationJMADataset(
        path_img_list=path_image_list,
        num_input_images=num_input_images,
        num_output_images=num_output_images,
        datetime_train_start=datetime_train_start,
        datetime_train_end=datetime_train_end,
        datetime_test_start=datetime_test_start,
        datetime_test_end=datetime_test_end,
        precipitation_threshold_train=0.2,
    )
    print(precipitation_jma_dataset[0][0].shape)
    print(precipitation_jma_dataset[1][0].shape)
