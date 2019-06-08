import os
import os.path as osp
import sys 
from PIL import Image
import six
import string
import lmdb
import pickle
import msgpack
from tqdm import tqdm
import pyarrow as pa
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets

class ImageFolderLMDB(data.Dataset):
    def __init__(self, dpath, split='train', transform=None, target_transform=None):
        # path = osp.join(db_path, split)
        db_path = osp.join(dpath,'%s.lmdb' % split)
        if not os.path.exists(db_path):
            print('No lmdb avialable, creating now.')
            folder2lmdb(dpath, name=split)
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.get(b'__len__')
            self.keys = msgpack.loads(txn.get(b'__keys__'))
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        img, label = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buff = six.BytesIO()
        buff.write(imgbuf)
        buff.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        label - unpacked[1]
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return img, label
    
    def __len__(self):
        return self.length
    
    def __repr__(self):
        return self.__class__.__name__ + '(' + self.db_path + ')'


def raw_reader(path):
    with open(path,'rb') as f:
        bin_data = f.read()
    return bin_data

def dumps_pyarrow(obj):
    """ Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()

def folder2lmdb(dpath, name='train', write_frequency=5000):
    """ Convert a folder to lmdb format.
    @Args:
    dpath: str, data path, this path must be an absoulte path
    name:  str, train/validation
    write_frequency: int, frequency to write the db 
    """
    path = osp.join(dpath, name)
    print('Loading dataset from %s'%path)
    dataset = ImageFolder(path, loader=raw_reader)
    data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x:x)

    lmdb_path = osp.join(dpath, "%s.lmdb"%name)
    isdir = osp.isdir(lmdb_path)
    
    print('Genetate LMDB to %s'%lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                    map_size=1099511627776 * 2, readonly=False,
                    meminit=False, map_async=True)
    
    txn = db.begin(write=True)
    for idx, data in enumerate(tqdm(data_loader)):
        image, label = data[0]
        txn.put(u'{}'.format(idx).encoder('ascii'), dumps_pyarrow((image, label)))
        if idx % write_frequency == 0:
            print('[%d/%d]'%(idx, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)
    # finish iterating 
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx+1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))
    
    print('Flushing dataset')
    db.sync()
    db.close()
    print('Finish writting.')

