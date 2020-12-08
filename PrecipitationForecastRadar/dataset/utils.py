from tqdm import tqdm
from multiprocessing import Pool
from typing import Optional, List, Tuple, Union, Callable, Dict
import os
import requests

import boto3
import botocore

S3_BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')


def argwrapper(args):
    return args[0](*args[1:])


def imap_unordered_bar(func, args, n_processes=15, extend=False, tqdm_disable=False,
                       init=None, credentials=None):
    """ execute for loop with multiprocessing

    Args:
        func (def): function of arg wrapper (argwrapper)
        args (list of args):  ex. [(target function, args of functions) for xx in xxxx]
        n_processes (int): number of processes
        django_process (bool): True for connection DB through django
        extend (bool): If True, each results will be merged by xxxx.extend(list)
        tqdm_disable (bool): If True, tqdm bar will not shown
        init (func): look def init(), call init(*credentials) before use this func
        credentials (list): [ftp_address, user name, password]

    Returns: appended or extended list

    """
    if (init is not None) and (credentials is not None):
        p = Pool(n_processes, initializer=init, initargs=credentials)
    else:
        p = Pool(n_processes)
    res_list = []
    with tqdm(total=len(args), disable=tqdm_disable) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            if extend:
                res_list.extend(res)
            else:
                res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def check_file_existence_local(path_file: str) -> bool:
    """ Check file existence and confirm the file is not empty

    Args:
        path_file (str): target file path

    Returns:
        (bool) If True, non-zero file exists in path_file

    """

    if os.path.exists(path_file) and os.path.getsize(path_file) > 0:
        print('file exist: ', path_file)
        return True
    return False


def check_file_existence_s3(path_file: str, multiprocessing: bool = False) -> bool:
    """ Check file existence and confirm the file is not empty

    Args:
        path_file (str): target file path
        multiprocessing (bool): True -> this code is included in the loop of multiprocessing.

    Returns:
        (bool) If True, non-zero file exists in path_file

    """
    if multiprocessing:
        session = boto3.session.Session()
        s3_client = session.client('s3')
    else:
        s3_client = boto3.client('s3')

    try:
        response = s3_client.head_object(
            Bucket=S3_BUCKET_NAME,
            Key=path_file
        )
        filesize = response['ContentLength']
    except botocore.exceptions.ClientError:
        print('file not exist in S3: ', path_file)
        return False

    if filesize > 0:
        return True
    else:
        return False


def get_s3_url_head(multiprocessing: bool = False) -> str:
    """ Get head of URL saved in S3

    Args:
        multiprocessing (bool): True -> this code is included in the loop of multiprocessing.

    Returns:
        (str): URL head

    """
    if multiprocessing:
        session = boto3.session.Session()
        s3 = session.client('s3')
    else:
        s3 = boto3.client('s3')

    bucket_location = s3.get_bucket_location(Bucket=S3_BUCKET_NAME)
    url_s3_head = "https://{0}.s3-{1}.amazonaws.com".format(
        S3_BUCKET_NAME,
        bucket_location['LocationConstraint']
    )
    return url_s3_head


def copy_to_s3(path_src_local: str, path_dst_s3: str, remove_local_file: bool = False, overwrite: bool = True,
               multiprocessing: bool = False) -> str:
    """ Copy local file to S3

    Args:
        path_src_local (str): source file on local
        path_dst_s3 (str): target path on S3
        remove_local_file (bool): If True, remove intermediate files after calculation.
        overwrite (bool): If True, overwrite existing files
        multiprocessing (bool): True -> this code is included in the loop of multiprocessing.

    Returns:
        (str): url of saved file on S3
    """

    assert os.path.exists(path_src_local), 'File not exist on local: {}'.format(path_src_local)

    if not overwrite and check_file_existence_s3(path_dst_s3, multiprocessing=multiprocessing):
        url_s3_head = get_s3_url_head(multiprocessing=multiprocessing)
        return os.path.join(url_s3_head, path_dst_s3)

    if multiprocessing:
        session = boto3.session.Session()
        s3_resource = session.resource('s3')
    else:
        s3_resource = boto3.resource('s3')

    bucket = s3_resource.Bucket(S3_BUCKET_NAME)

    bucket.upload_file(path_src_local, path_dst_s3)
    url_s3_head = get_s3_url_head(multiprocessing=multiprocessing)

    if remove_local_file:
        os.remove(path_src_local)

    return os.path.join(url_s3_head, path_dst_s3)


# def get_all_file_path_s3(dir_parent: str, ext_filter: Optional[Union[str, List[str]]] = None) -> List[str]:
#     """ Get all of files' paths under specified directory on S3
#     Args:
#         dir_parent (str): parent directory for searching paths
#         ext_filter (list): list of string for selecting files which are matched from end of the filenames. If None, all of the files are returned
#     Returns:
#         (str) list of paths
#     """
#     s3_resource = boto3.resource('s3')
#     my_bucket = s3_resource.Bucket(S3_BUCKET_NAME)
#     objects = my_bucket.objects.filter(Prefix=dir_parent)
#     url_head = get_s3_url_head()
#     if ext_filter is None:
#         path_out = [os.path.join(url_head, obj.key) for obj in objects]
#     else:
#         if type(ext_filter) != list:
#             ext_filter = list(ext_filter)
#
#         path_out = [os.path.join(url_head, obj.key) for ext_elem in ext_filter for obj in objects if
#                     obj.key.endswith(ext_elem)]
#     return path_out

def get_all_file_path_s3(dir_parent: str, ext_filter: Optional[Union[str, List[str]]] = None,
                         func_kwargs: Optional[Union[Callable, Tuple[Callable, Dict]]] = None
                         ) -> List[str]:
    """ Get all of files' paths under specified directory on S3
    Args:
        dir_parent (str): parent directory for searching paths
        ext_filter (list): list of string for selecting files which are matched from end of the filenames. If None, all of the files are returned
        func_kwargs (Optional[Union[Callable, Tuple[Callable, Dict]]]):
            function of filtering, with args in dictionary
    Returns:
        (str) list of paths
    """
    s3_resource = boto3.resource('s3')
    my_bucket = s3_resource.Bucket(S3_BUCKET_NAME)
    objects = my_bucket.objects.filter(Prefix=dir_parent)
    url_head = get_s3_url_head()
    if ext_filter is None:
        path_out = [os.path.join(url_head, obj.key) for obj in objects]
    else:
        if type(ext_filter) != list:
            ext_filter = [ext_filter]

        path_out = []
        for obj in objects:
            if func_kwargs is not None:
                if type(func_kwargs) == tuple and (not func_kwargs[0](obj.key, **func_kwargs[1])):
                    continue
                elif type(func_kwargs) != tuple and not func_kwargs(obj.key):
                    continue

            for ext_elem in ext_filter:
                if obj.key.endswith(ext_elem):
                    path_out.append(os.path.join(url_head, obj.key))

        # path_out = [os.path.join(url_head, obj.key) for ext_elem in ext_filter for obj in objects if
        #             obj.key.endswith(ext_elem)]
    return path_out


def download_from_http(url_src: str, dir_dst: str, filename: Optional[str] = None, overwrite: bool = True) -> str:
    """ Download file from target url to local under a specific directory
    
    Args:
        url_src (str): target file's url 
        dir_dst (str): directory path of destination 
        filename (Optional[str]): If None, filename is copied from url_src
        overwrite (bool): If True, avoid downloading the file with the same name and not empty 

    Returns:
        (str) local path of downloaded file

    """

    if filename is None:
        filename = os.path.basename(url_src)

    path_dst = os.path.join(dir_dst, filename)

    if overwrite or not check_file_existence_local(path_dst):
        r = requests.get(url_src, stream=True)
        if r.status_code == requests.codes.ok:
            if not os.path.exists(dir_dst):
                os.makedirs(dir_dst)
            with open(path_dst, 'wb') as f:
                for data in r:
                    f.write(data)
            return path_dst
        else:
            return ''
    else:
        print('keep existing file: ', path_dst)
        return path_dst
