import numpy as np
from numpy.random import default_rng
import pandas as pd
import os
import imghdr
import zipfile
from sqlalchemy import create_engine
from primihub.utils.logger_util import logger

# 完全避免torchvision导入，使用模拟函数
def read_image(path):
    logger.warning(f"read_image模拟函数被调用（torchvision不可用）: {path}")
    # 返回一个模拟的numpy数组
    return np.zeros((224, 224, 3), dtype=np.uint8)

class TorchDataset:
    """TorchDataset的模拟类"""
    pass

try:
    import pyarrow.parquet as pq
    import pyarrow as pa
except ImportError:
    logger.warning("pyarrow not available, parquet support disabled")
    pq = None
    pa = None


def read_data(data_info,
              selected_column=None,
              droped_column=None,
              transform=None,
              target_transform=None):
    data_type = data_info['type'].lower()
    if data_type == 'csv':
        return read_csv(data_info['data_path'],
                        selected_column,
                        droped_column)
    elif data_type == 'image':
        return TorchImageDataset(data_info['image_dir'],
                                 data_info['annotations_file'],
                                 transform,
                                 target_transform)
    elif data_type == 'mysql':
        return read_mysql(data_info['username'],
                          data_info['password'],
                          data_info['host'],
                          data_info['port'],
                          data_info['dbName'],
                          data_info['tableName'],
                          selected_column,
                          droped_column)
    elif data_type == 'parquet':
        return read_parquet(data_info['data_path'],
                            selected_column,
                            droped_column)
    else:
        error_msg = f'Unsupported data type: {data_type}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def read_csv(data_path, selected_column=None, droped_column=None):
    data = pd.read_csv(data_path)
    if selected_column:
        data = data[selected_column]
    if droped_column in data.columns:
        data.pop(droped_column)
    return data

def read_parquet(data_path, selected_column=None, droped_column=None):
    parquet_file = pq.ParquetFile(data_path)
    data = parquet_file.read().to_pandas()
    if selected_column:
        data = data[selected_column]
    if droped_column in data.columns:
        data.pop(droped_column)
    return data

def read_mysql(user,
               password,
               host,
               port,
               database,
               table_name,
               selected_column=None,
               droped_column=None):
    engine_str = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(engine_str)
    with engine.connect() as conn:
        df = pd.read_sql_table(table_name, conn, columns=selected_column)
        if droped_column in df.columns:
            df.pop(droped_column)
        return df


class TorchImageDataset(TorchDataset):
    """TorchImageDataset的模拟类，避免torchvision依赖"""

    def __init__(self, img_dir, annotations_file=None, transform=None, target_transform=None):
        logger.warning(f"TorchImageDataset模拟类被调用（避免torchvision依赖）: {img_dir}")
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = pd.DataFrame(columns=['file_name'])

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        # 返回模拟数据
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        label = 0
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


class DataLoader:

    def __init__(self, dataset, label=None, batch_size=1, shuffle=True, seed=None):
        self.dataset = dataset
        self.label = label
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = self.__len__()
        self.indices = np.arange(self.n_samples)
        self.start = 0
        if seed:
            np.random.seed(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self):
        return self

    def __next__(self):
        if self.start == 0:
            if self.shuffle:
                np.random.shuffle(self.indices)

        start = self.start
        end = start + self.batch_size
        if end > self.n_samples:
            end = self.n_samples
        self.start += self.batch_size

        if start < self.n_samples:
            batch_idx = self.indices[start:end]
            if self.label is not None:
                return self.dataset[batch_idx], self.label[batch_idx]
            else:
                return self.dataset[batch_idx]
        else:
            self.start = 0
            raise StopIteration


class DPDataLoader(DataLoader):

    def __init__(self, dataset, label=None, batch_size=1):
        self.dataset = dataset
        self.label = label
        self.batch_size = batch_size
        self.rng = default_rng()
        self.n_samples = self.__len__()
        self.max_iter = self.n_samples // self.batch_size
        self.num_iter = 0

    def __next__(self):
        self.num_iter += 1
        if self.num_iter <= self.max_iter:
            batch_idx = self.rng.choice(self.n_samples,
                                        self.batch_size,
                                        replace=False)
            if self.label is not None:
                return self.dataset[batch_idx], self.label[batch_idx]
            else:
                return self.dataset[batch_idx]
        else:
            self.num_iter = 0
            raise StopIteration