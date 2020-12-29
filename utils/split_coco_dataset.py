import json
import funcy
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from collections import defaultdict, Counter
from utils.dataset_converter import concatenate_datasets
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# filter_annotations and save_coco on akarazniewicz/cocosplit
def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda im: int(im['id']), images)
    return funcy.lfilter(lambda ann:
                         int(ann['image_id']) in image_ids, annotations)


def save_coco(dest, info, licenses,
              images, annotations, categories):
    with open(dest, 'w') as f:
        json.dump({'info': info,
                   'licenses': licenses,
                   'images': images,
                   'annotations': annotations,
                   'categories': categories},
                  f, indent=2, sort_keys=True)


def PseudoStratifiedShuffleSplit(images,
                                 annotations,
                                 test_size):
    # count categories per image
    categories_per_image = defaultdict(Counter)
    for ann in annotations:
        categories_per_image[ann['image_id']][ann['category_id']] += 1

    # find category with most annotations per image
    max_category = []
    for cat in categories_per_image.values():
        cat = cat.most_common(1)[0][0]
        max_category.append(cat)

    # pseudo-stratified-split
    strat_split = StratifiedShuffleSplit(n_splits=1,
                                         test_size=test_size,
                                         random_state=2020)

    for train_index, test_index in strat_split.split(images,
                                                     np.array(max_category)):
        x = [images[i] for i in train_index]
        y = [images[i] for i in test_index]
    print('Train:', len(x), 'images, valid:', len(y))
    return x, y


# function based on https://github.com/trent-b/iterative-stratification'''
def MultiStratifiedShuffleSplit(images,
                                annotations,
                                test_size):
    # count categories per image
    categories_per_image = defaultdict(Counter)
    max_id = 0
    for ann in annotations:
        categories_per_image[ann['image_id']][ann['category_id']] += 1
        if ann['category_id'] > max_id:
            max_id = ann['category_id']

    # prepare list with count of cateory objects per image
    all_categories = []
    for cat in categories_per_image.values():
        pair = []
        for i in range(1, max_id + 1):
            pair.append(cat[i])
        all_categories.append(pair)

    # multilabel-stratified-split
    strat_split = MultilabelStratifiedShuffleSplit(n_splits=1,
                                                   test_size=test_size,
                                                   random_state=2020)

    for train_index, test_index in strat_split.split(images,
                                                     all_categories):
        x = [images[i] for i in train_index]
        y = [images[i] for i in test_index]
    print('Train:', len(x), 'images, valid:', len(y))
    return x, y


# split_coco_dataset partially based on akarazniewicz/cocosplit
def split_coco_dataset(list_of_datasets_to_split,
                       dest,
                       test_size=0.2,
                       mode='multi'):
    if len(list_of_datasets_to_split) > 1:
        dataset = concatenate_datasets(list_of_datasets_to_split)
    else:
        with open(list_of_datasets_to_split[0], 'r') as f:
            dataset = json.loads(f.read())

    categories = dataset['categories']
    info = dataset['info']
    licenses = dataset['licenses']
    annotations = dataset['annotations']
    images = dataset['images']

    images_with_annotations = funcy.lmap(lambda ann:
                                         int(ann['image_id']), annotations)
    images = funcy.lremove(lambda i: i['id'] not in
                           images_with_annotations, images)
    if mode == 'multi':
        x, y = MultiStratifiedShuffleSplit(images, annotations, test_size)
    else:
        x, y = PseudoStratifiedShuffleSplit(images, annotations, test_size)

    save_coco(dest+'_train.json', info, licenses,
              x, filter_annotations(annotations, x), categories)
    save_coco(dest+'_test.json', info, licenses, y,
              filter_annotations(annotations, y), categories)

    print('Finished stratified shuffle split. Results saved in:',
          dest + '_train.json', dest + '_test.json')
