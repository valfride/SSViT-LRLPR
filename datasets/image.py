import lmdb
import pickle
import cv2
import numpy as np
from datasets import register
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

@register('paired_images')
class PairedImages(Dataset):
    def __init__(self, path_split, phase='training'):

        self.split_file = Path(path_split)

        self.phase =  phase
        self.dataset = []
        
        with open(self.split_file, 'r') as f:
            data = f.readlines()
        
        for path in data:
            hr, lr, split = path.split(';')
            sample = {'hr': hr,
                    'lr': lr   
                    }            
            
            if self.phase in split:
                self.dataset.append(sample)
                    
    def __len__(self):
            return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]

@register('parallel_training')
class parallel_training(Dataset):
    def __init__(self, path_split, phase='training'):
        self.split_file = path_split
        self.phase = phase
        self.dataset = []
        
        with open(self.split_file, 'r') as f:
            data = f.readlines()    
        for path in data:    
            path_hr, path_lr, split = path.split(';')
            
            path_hr = path_hr.strip()
            path_lr = path_lr.strip()
            
            sample = {"lr": path_lr,
                    "hr": path_hr
                    }
            
            if self.phase in split:
                self.dataset.append(sample)
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]

@register('ocr_img')
class ocr_dataset(Dataset):
    def __init__(self, path_split, phase='training'):
        self.split_file = path_split
        self.phase = phase
        self.dataset = []
        
        with open(self.split_file, 'r') as f:
            data = f.readlines()    
        for path in data:    
            _, path_imgs, split = path.split(';')
            path_imgs = path_imgs.strip()
            # if '001.png' in path_imgs:
            sample = {"img": path_imgs,
                    }
        
            if self.phase in split:
                self.dataset.append(sample)
            else:
                pass
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset[index]


@register('multi_image')
class multi_image(Dataset):
    def __init__(self, path_split, phase='training'):

        self.split_file = path_split
        self.phase =  phase
        self.dataset = []
        
        with open(self.split_file, 'r') as f:
            data = f.readlines()
        
        for path in data:
            gt, path_imgs, split = path.split(';')
            path_imgs = path_imgs.strip()
            sample = {'gt': gt,
                    'imgs': Path(path_imgs)   
                    }    
            
            if self.phase in split:
                self.dataset.append(sample)
                    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return self.dataset[index]
        
            
@register('lmdb_dataset')
class LMDBDataset(Dataset):
    def __init__(self, path_split, phase='training', in_memory=False):
        """
        Args:
            path_split: Path to the LMDB directory.
            phase: 'training' or 'validation'.
            in_memory: If True, loads the ENTIRE dataset bytes into RAM during init.
        """
        self.lmdb_dir = path_split
        self.phase = phase
        self.in_memory = in_memory
        
        # 1. Load the metadata pickle file
        metadata_path = Path(self.lmdb_dir) / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            all_metadata = pickle.load(f)
        
        # 2. Filter for the requested phase
        self.dataset = [item for item in all_metadata if self.phase in item['split']]
        
        self.env = None
        self.ram_cache = {} # This will hold the bytes if in_memory=True

        # 3. Explicit RAM Caching
        if self.in_memory:
            print(f"🔥 CACHING {phase.upper()} DB TO RAM: Loading {len(self.dataset)} images...")
            # Open a temporary env just for the initial load
            temp_env = lmdb.open(self.lmdb_dir, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
            with temp_env.begin(write=False) as txn:
                for item in tqdm(self.dataset, desc=f"Loading {phase} bytes"):
                    key = item['lmdb_key']
                    # We store the raw bytes in the dictionary
                    self.ram_cache[key] = txn.get(key)
            temp_env.close()
            print("✅ Caching complete!")

    def _init_env(self):
        # We only need the LMDB environment if we are NOT using the RAM cache
        if self.env is None and not self.in_memory:
            self.env = lmdb.open(self.lmdb_dir, max_readers=64, readonly=True, lock=False, readahead=False, meminit=False)
            
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        key = item['lmdb_key']
        
        # 1. Fetch the raw compressed bytes (From RAM or LMDB)
        if self.in_memory:
            # INSTANT access from the Python dictionary
            image_bytes = self.ram_cache[key]
        else:
            # Fallback to reading from the LMDB file
            self._init_env()
            with self.env.begin(write=False) as txn:
                image_bytes = txn.get(key)
            
        # 2. Decode the bytes into a Numpy array (CPU task)
        img_array = np.frombuffer(image_bytes, dtype=np.uint8)
        img_raw = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        
        return {
            'img_raw': img_raw,
            'gt': item['gt'],
            'name': Path(item['original_path']).name,
            'img_path': item['original_path']
        }
