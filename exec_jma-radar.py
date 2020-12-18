import datetime, pytz
from PrecipitationForecastRadar.dataset.jma_radar import DatasetMaker

if __name__=='__main__':
    dir_parent_src = 'data/JMA/RA/converted' #'data/JMA/RA/converted/RA2015'
    # lon = 127.8
    # lat = 26.3
    lon = 140.662799
    lat = 35.954808
    size = (256, 256)

    datetime_start = datetime.datetime(year=2015, month=1, day=1, hour=0, minute=0, tzinfo=pytz.utc)
    datetime_end = datetime.datetime(year=2016, month=12, day=31, hour=23, minute=59, tzinfo=pytz.utc)
    dir_parent_dst_local = 'dataset/RA'
    dir_parent_dst_s3 = 'dataset/RA'
    subdir_dst = '2015-2016_kashima'
    src_s3 = True
    overwrite = True
    s3_upload = True
    remove_local_file = False
    processes = 15

    dataset_maker = DatasetMaker(dir_parent_src=dir_parent_src, dir_parent_dst_local=dir_parent_dst_local,
                                 dir_parent_dst_s3=dir_parent_dst_s3, subdir_dst=subdir_dst, src_is_s3=src_s3)
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
