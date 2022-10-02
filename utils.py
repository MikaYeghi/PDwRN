import os
import json
from tqdm import tqdm

from detectron2.structures import BoxMode


def register_dataset(data_path, mode, debug_on=False):
    """
    This function registers a new dataset. In order to do this, it takes as inputs some parameters which describe the dataset,
    and transforms them into the dataset dictionary format expected by the model.
    """
    anns_dir = "annotations"
    imgs_dir = "train2017" if mode == "train" else "val2017"
    annotations_file = os.path.join(data_path, anns_dir, "instances_train2017.json")
    
    # Initialize the return list where the dicts will be stored
    dataset_dicts = []
    
    # Load the annotations file
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create the category mapping from COCO to ordered id-s
    category_map = category_mapping(annotations['categories'])
    
    # Loop through the images
    full_annotations = annotations['images'][:100] if debug_on else annotations['images']
    for idx, image in enumerate(tqdm(full_annotations)):
        record = {}
        
        # Record preliminary information about the image
        record['file_name'] = os.path.join(data_path, imgs_dir, image['file_name'])
        record['image_id'] = idx
        record['height'] = image['height']
        record['width'] = image['width']
        
        # Record the detections (into objs)
        objs = []
        detections = [ann for ann in annotations['annotations'] if ann['image_id']==image['id']]
        for detection in detections:
            obj = {
                "bbox": detection['bbox'],
                "bbox_mode": BoxMode.XYWH_REL,
                "category_id": category_map[detection['category_id']]
            }
            objs.append(obj)
        
        # Record the detections
        record['annotations'] = objs
        
        # Add the image description to the dataset
        dataset_dicts.append(record)
    
    return dataset_dicts

def category_mapping(categories):
    """
    This function takes COCO format category descriptions and maps them to new, ordered categories.
    The reason for implementing this function is that COCO misses some category id-s. For example, it doesn't have
    a category id 12.
    """
    k = 0
    category_map = {}
    for category in categories:
        category_map[category['id']] = k
        k += 1
    return category_map

def get_category_names(anns_file_path):
    # Load the annotations file
    with open(anns_file_path, 'r') as f:
        annotations = json.load(f)
    
    category_names = []
    for category in annotations['categories']:
        category_names.append(category['name'])
    
    return category_names