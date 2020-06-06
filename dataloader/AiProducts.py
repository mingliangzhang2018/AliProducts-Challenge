import random, cv2
import torch
import json, os, random, time
import cv2
import torchvision.transforms as transforms
from dataloader.transform_wrapper import TRANSFORMS
import numpy as np
from torch.utils.data import Dataset


class BaseSet(Dataset):
    def __init__(self, mode="train", cfg=None, transform=None):
        self.mode = mode
        self.transform = transform
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE
        self.size = self.input_size
        self.data_root = cfg.DATASET_ROOT
        print("Use {} Mode to train network".format(self.color_space))

        if self.mode == "train":
            print("Loading train data ...", end=" ")
            self.json_path = cfg.DATASET_TRAIN_JSON
        elif "valid" in self.mode:
            print("Loading valid data ...", end=" ")
            self.json_path = cfg.DATASET_VALID_JSON
        elif "test" in self.mode:
            print("Loading test data ...", end=" ")
            self.json_path = cfg.DATASET_TEST_JSON
        else:
            raise NotImplementedError
        self.update_transform()

        with open(self.json_path, "r") as f:
            self.all_info = json.load(f)
        if "test" in self.mode:
            self.num_classes = cfg.num_classes
            self.data = self.all_info["images"]
        else:
            self.num_classes = self.all_info["num_classes"]
            self.data = self.all_info["annotations"]
        print("Contain {} images of {} classes".format(len(self.data),
                                                       self.num_classes))

    def update_transform(self, input_size=None):

        normalize = TRANSFORMS["normalize"](cfg=self.cfg,
                                            input_size=input_size)

        transform_list = [transforms.ToPILImage()]

        transform_ops = (("random_resized_crop", "random_horizontal_flip")
                         if self.mode == "train" else
                         ("shorter_resize_for_crop", "center_crop"))

        for tran in transform_ops:
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg,
                                                   input_size=input_size))

        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list)

    def get_num_classes(self):
        return self.num_classes

    def get_annotations(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def imread_with_retry(self, fpath):
        retry_time = 10
        for k in range(retry_time):
            try:
                img = cv2.imread(fpath)
                if img is None:
                    print("img is None, try to re-read img")
                    continue
                return img
            except Exception as e:
                if k == retry_time - 1:
                    assert False, "cv2 imread {} failed".format(fpath)
                time.sleep(0.1)

    def _get_image(self, now_info):
        #if self.data_type == "jpg":
        if "test" in self.mode:
            fpath = os.path.join(self.data_root, now_info["image_id"])
        else:
            fpath = os.path.join(self.data_root, now_info["fpath"])

        img = self.imread_with_retry(fpath)
        if self.color_space == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _get_class_dict(self):
        class_dict = dict()
        for i, anno in enumerate(self.data):
            cat_id = (anno["category_id"]
                      if "category_id" in anno else anno["image_label"])
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)
        return class_dict

    def get_weight(self, annotations, num_classes):
        num_list = [0] * num_classes
        cat_list = []
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1
            cat_list.append(category_id)
        max_num = max(num_list)
        class_weight = [max_num / i for i in num_list]
        sum_weight = sum(class_weight)
        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        rand_number, now_sum = random.random() * self.sum_weight, 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i


class AiProducts(BaseSet):
    def __init__(self, mode='train', cfg=None, transform=None):
        super().__init__(mode, cfg, transform)
        random.seed(0)
        if self.mode == "train" and not self.cfg.SAMPLER_TYPE == "instance_balance":
            self.class_weight, self.sum_weight = self.get_weight(
                self.data, self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):

        if self.mode == "train":
            assert self.cfg.SAMPLER_TYPE in [
                "instance_balance", "class_balance", "reverse"
            ]
            if self.cfg.SAMPLER_TYPE == "instance_balance":
                pass
            if self.cfg.SAMPLER_TYPE == "class_balance":
                sample_class = random.randint(0, self.cfg.num_classes - 1)
                sample_indexes = self.class_dict[sample_class]
                index = random.choice(sample_indexes)
            if self.cfg.SAMPLER_TYPE == "reverse":
                sample_class = self.sample_class_index_by_weight()
                sample_indexes = self.class_dict[sample_class]
                index = random.choice(sample_indexes)

        now_info = self.data[index]
        img = self._get_image(now_info)
        image = self.transform(img)
        if "test" in self.mode:
            return image, now_info["image_id"]
        else:
            image_label = now_info['category_id']  # 0-index
            return image, image_label
