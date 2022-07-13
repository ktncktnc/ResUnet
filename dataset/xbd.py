import os
import json
import cv2
import numpy as np
import random
import albumentations as A
from utils.augmentation import *
from albumentations.pytorch import ToTensorV2
from shapely import wkt
from shapely.geometry import Polygon
import torch
from torch.utils.data import Dataset, DataLoader
import multiprocessing

multiprocessing.set_start_method('spawn', True)

holdout_train = ["guatemala-volcano", "hurricane-florence", "hurricane-harvey", "hurricane-matthew",
                 "hurricane-michael", "mexico-earthquake", "midwest-flooding", "palu-tsunami", "santa-rosa-wildfire",
                 "socal-fire", "lower-puna-volcano", "moore-tornado", "nepal-flooding", "portugal-wildfire",
                 "tuscaloosa-tornado", "woolsey-fire"]
holdout_test = ["joplin-tornado", "pinery-bushfire", "sunda-tsunami"]
holdout2_train = ["guatemala-volcano", "hurricane-florence", "hurricane-harvey", "hurricane-matthew",
                  "hurricane-michael", "mexico-earthquake", "midwest-flooding", "palu-tsunami", "santa-rosa-wildfire",
                  "socal-fire", "nepal-flooding", "joplin-tornado", "pinery-bushfire", "sunda-tsunami",
                  "tuscaloosa-tornado", "lower-puna-volcano", "woolsey-fire"]
holdout2_test = ["moore-tornado", "portugal-wildfire"]
holdout3_train = ["guatemala-volcano", "hurricane-florence", "hurricane-harvey", "hurricane-matthew",
                  "hurricane-michael", "mexico-earthquake", "midwest-flooding", "palu-tsunami", "santa-rosa-wildfire",
                  "socal-fire", "nepal-flooding", "joplin-tornado", "pinery-bushfire", "portugal-wildfire",
                  "moore-tornado", "sunda-tsunami"]
holdout3_test = ["tuscaloosa-tornado", "lower-puna-volcano", "woolsey-fire"]
gupta_train = ["guatemala-volcano", "hurricane-michael", "santa-rosa-wildfire", "hurricane-florence",
               "midwest-flooding", "palu-tsunami", "socal-fire", "hurricane-harvey", "mexico-earthquake",
               "hurricane-matthew", "nepal-flooding"]
gupta_test = ["tuscaloosa-tornado", "lower-puna-volcano", "woolsey-fire", "joplin-tornado", "pinery-bushfire",
              "portugal-wildfire", "moore-tornado", "sunda-tsunami"]


class XView2Dataset(torch.utils.data.Dataset):
    """xView2
    input: Post image
    target: pixel-wise classes
    """
    dmg_type = {'background': 0, 'no-damage': 1, 'minor-damage': 2, 'major-damage': 3, 'destroyed': 4,
                'un-classified': 255}
    diaster_type = {'earthquake': 0, 'fire': 1, 'tsunami': 2, 'volcano': 3, 'wind': 4, 'flooding': 5}

    def __init__(self, root_dir, with_mask=True, resized_shape=(256, 256), rgb_bgr='rgb', preprocessing=None, mode='train', divide=2,
                 single_disaster=None, augment_transform=None):
        # assert mode in ('train', 'test', 'oodtrain', 'oodtest', 'oodhold',"guptatrain","guptahold")
        self.mode = mode
        self.root = root_dir
        self.with_mask = with_mask
        self.divide = divide
        self.resized_shape = resized_shape
        self.divide_width = self.divide_height = 512
        self.width = self.height = 1024

        assert rgb_bgr in ('rgb', 'bgr')
        self.rgb = bool(rgb_bgr == 'rgb')
        self.preprocessing = preprocessing
        self.dirs = {'train_imgs': os.path.join(self.root, 'train', 'images'),
                     'train_labs': os.path.join(self.root, 'train', 'targets'),
                     'tier3_imgs': os.path.join(self.root, 'tier3', 'images'),
                     'tier3_labs': os.path.join(self.root, 'tier3', 'targets'),
                     'test_imgs': os.path.join(self.root, 'test', 'images'),
                     'test_labs': os.path.join(self.root, 'test', 'targets'),
                     'hold_imgs': os.path.join(self.root, 'hold', 'images'),
                     'hold_labs': os.path.join(self.root, 'hold', 'targets')}
        train_imgs = [s for s in os.listdir(self.dirs['train_imgs'])]
        tier3_imgs = [s for s in os.listdir(self.dirs['tier3_imgs'])]
        test_imgs = [s for s in os.listdir(self.dirs['test_imgs'])]
        hold_imgs = [s for s in os.listdir(self.dirs['hold_imgs'])]

        if self.with_mask:
            train_labs = [s for s in os.listdir(self.dirs['train_labs'])]
            tier3_labs = [s for s in os.listdir(self.dirs['tier3_labs'])]
            test_labs = [s for s in os.listdir(self.dirs['test_labs'])]
            hold_labs = [s for s in os.listdir(self.dirs['hold_labs'])]
        else:
            train_labs = None
            tier3_labs = None
            test_labs = None
            hold_labs = None

        self.sample_files = []
        if self.mode == 'train':
            self.add_samples_train(self.dirs['train_imgs'], self.dirs['train_labs'], train_imgs, train_labs)
        elif self.mode == 'train_tier3':
            self.add_samples_train(self.dirs['train_imgs'], self.dirs['train_labs'], train_imgs, train_labs)
            self.add_samples_train(self.dirs['tier3_imgs'], self.dirs['tier3_labs'], tier3_imgs, tier3_labs)
        elif self.mode in ["test"]:
            self.add_samples_train(self.dirs['test_imgs'], self.dirs['test_labs'], test_imgs, test_labs)

        self.random_crop = RandomCropSaveSegmentMask(512, 512)
        if augment_transform is None:
            self.data_transforms = self.get_default_transform(mode)
        else:
            self.data_transforms = augment_transform

    def get_default_transform(self, mode):
        if mode not in ['test', 'oodtest', 'oodhold', 'guptatest', 'guptahold', "ood2test", "ood2hold",
                        "ood3test", "ood3hold", "singletest", "singlehold"]:
            return A.Compose([
                A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.5, 0.1), rotate_limit=10),
                A.RandomRotate90(),
                A.RandomGamma(),
                A.Blur(blur_limit=5, p=0.4),
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
                A.Resize(self.resized_shape[0], self.resized_shape[1]),
                PerImageStandazation(),
                ToTensorV2()
            ],
                additional_targets={
                    'post_image': 'image',
                }
            )
        else:
            return A.Compose([
                A.Resize(self.resized_shape[0], self.resized_shape[1]),
                PerImageStandazation(),
                ToTensorV2()
            ],
                additional_targets={
                    'post_image': 'image',
                }
            )

    def add_samples_train(self, img_dirs, lab_dirs, imgs, labs=None):
        for pre in os.listdir(img_dirs):
            if pre[-17:] != '_pre_disaster.png':
                continue
            chop = pre[:-4].split('_')
            disaster = chop[0]
            img_id = '_'.join(chop[:2])
            post = img_id + '_post_disaster.png'
            pre_label = img_id + '_pre_disaster_target.png'
            post_label = img_id + '_post_disaster_target.png'
            assert post in imgs
            if self.with_mask:
                assert pre_label in labs
                assert post_label in labs

            assert img_id not in self.sample_files
            files = {
                'img_id': img_id,
                 'pre_img': os.path.join(img_dirs, pre),
                 'post_img': os.path.join(img_dirs, post),
                 'pre_label': os.path.join(lab_dirs, pre_label),
                 'post_label': os.path.join(lab_dirs, post_label)
            }
            self.sample_files += [
                {**files, **{'divide': i}} for i in range(self.divide*self.divide)
            ]

    def get_sample_info(self, idx):
        files = self.sample_files[idx]
        pre_img = cv2.imread(files['pre_img'])
        post_img = cv2.imread(files['post_img'])

        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

        pre_json = json.loads(open(files['pre_json']).read())
        post_json = json.loads(open(files['post_json']).read())

        sample = {'pre_img': pre_img, 'post_img': post_img, 'image_id': files['img_id'],
                  'im_width': post_json['metadata']['width'],
                  'im_height': post_json['metadata']['height'],
                  'disaster': post_json['metadata']['disaster_type'],
                  'pre_meta': {m: pre_json['metadata'][m] for m in pre_json['metadata']},
                  'post_meta': {m: post_json['metadata'][m] for m in post_json['metadata']},
                  'pre_builds': dict(), 'post_builds': dict(), 'builds': dict()}
        for b in pre_json['features']['xy']:
            buid = b['properties']['uid']
            sample['pre_builds'][buid] = {p: b['properties'][p] for p in b['properties']}
            poly = Polygon(wkt.loads(b['wkt']))
            sample['pre_builds'][buid]['poly'] = list(poly.exterior.coords)
        for b in post_json['features']['xy']:
            buid = b['properties']['uid']
            sample['post_builds'][buid] = {p: b['properties'][p] for p in b['properties']}
            poly = Polygon(wkt.loads(b['wkt']))
            sample['post_builds'][buid]['poly'] = list(poly.exterior.coords)
            sample['builds'][buid] = {'poly': list(poly.exterior.coords),
                                      'subtype': b['properties']['subtype']}
        # sample['mask_img'] = self.make_mask_img(**sample)
        return sample

    def __getitem__(self, idx):
        files = self.sample_files[idx]
        x1, x2, y1, y2 = self.get_coord(files['divide'])

        pre_img = cv2.imread(files['pre_img'])[x1:x2, y1:y2, ...]
        post_img = cv2.imread(files['post_img'])[x1:x2, y1:y2, ...]
        sample = {
            'image': pre_img,
            'post_image': post_img
        }

        if self.with_mask:
            pre_label = cv2.imread(files['pre_label'])[x1:x2, y1:y2, 0]
            post_label = cv2.imread(files['post_label'])[x1:x2, y1:y2, 0]
            masks = np.zeros((post_img.shape[0], post_img.shape[1], 5))
            masks[:, :, 0] = (pre_label == 0) * 1
            for i in range(1, 5):
                masks[:, :, i] = (post_label == i) * 1

            sample['masks'] = [masks[:, :, i] for i in range(5)]
        # self.random_crop.get_best_patch(pre_label)

        transformed = self.data_transforms(**sample)
        image1 = transformed['image']
        image2 = transformed['post_image']

        if self.with_mask:
            masks = transformed['masks']
            masks = torch.FloatTensor(masks)
            return dict(x=image1.float(), y=image2.float(), masks=masks.float())
        return dict(x=image1.float(), y=image2.float())

    @staticmethod
    def _get_building_from_json(post_json):
        buildings = dict()
        for b in post_json['features']['xy']:
            buid = b['properties']['uid']
            poly = Polygon(wkt.loads(b['wkt']))
            buildings[buid] = {'poly': list(poly.exterior.coords),
                               'subtype': b['properties']['subtype']}
        return buildings

    def get_sample_with_mask(self, files, pre_img, post_img):
        post_json = json.loads(open(files['post_json']).read())
        sample = {'pre_img': pre_img, 'post_img': post_img, 'image_id': files['img_id'],
                  'disaster': self.diaster_type[post_json['metadata']['disaster_type']]}

        buildings = self._get_building_from_json(post_json)
        sample['mask_img'] = self.make_mask_img(**buildings)
        return sample

    def make_mask_img(self, **kwargs):
        width = 1024
        height = 1024
        builings = kwargs

        mask_img = np.zeros([height, width], dtype=np.uint8)
        for dmg in self.dmg_type:
            polys_dmg = [np.array(builings[p]['poly']).round().astype(np.int32).reshape(-1, 1, 2)
                         for p in builings if builings[p]['subtype'] == dmg]
            cv2.fillPoly(mask_img, polys_dmg, [self.dmg_type[dmg]])

        return mask_img

    def show_sample(self, **kwargs):
        pass

    def __len__(self):
        return len(self.sample_files)

    def get_coord(self, divide):
        row = int(divide/self.divide)
        col = divide - row*self.divide

        x1 = self.divide_height*row
        if row < self.divide - 1:
            x2 = self.divide_height*(row + 1)
        else:
            x2 = self.height

        y1 = self.divide_width*col
        if col < self.divide - 1:
            y2 = self.divide_width*(col + 1)
        else:
            y2 = self.width

        return x1, x2, y1, y2

if __name__ == '__main__':
    root_path = "/mnt/Dataset/xView2/v2"
    dataset = XView2Dataset(root_path, rgb_bgr='rgb', preprocessing={'flip': True, 'scale': None, 'crop': (513, 513)})
    dataset_test = XView2Dataset(root_path, rgb_bgr='rgb',
                                 preprocessing={'flip': False, 'scale': (0.8, 2.0), 'crop': (1024, 1024)})

    n_samples = len(dataset)
    n_train = int(n_samples * 0.85)
    n_test = n_samples - n_train
    trainset, testset = torch.utils.data.random_split(dataset, [n_train, n_test])

    dataloader = DataLoader(trainset, batch_size=5, shuffle=True, num_workers=4)

    for i in range(n_test):
        sample = testset[i]
        original_idx = testset.indices[i]
        info = dataset.get_sample_info(original_idx)
        info2 = dataset_test.get_sample_info(original_idx)
        sample2 = dataset_test[original_idx]
        print(i, original_idx, sample['disaster'], sample['image_id'], sample['post_img'].shape)
        print(i, original_idx, sample2['disaster'], sample2['image_id'], sample2['post_img'].shape)
        print(i, original_idx, info['disaster'], info['image_id'])
        print(i, original_idx, info2['disaster'], info2['image_id'])

    for i, samples in enumerate(dataloader):
        print(i, samples['disaster'], samples['image_id'], samples['post_img'].shape)
