import os
import pandas
from pathlib import Path
from ipywidgets import widgets, Layout
from IPython.display import clear_output, display
import torch
import numpy as np


def create_csv_with_image_paths(csv_path, csv_image_column, image_folder, attributes=None, image_extension='.jpg'):
    """creates a new csv at the csv_path with image_column filled with the relative path from the <b>image_folder</b> and empty columns <strong>attributes</strong>"""
    columns = [csv_image_column]
    if attributes is not None:
        if type(attributes) is type(dict):
            attributes = list(attributes.keys())
        columns+=attributes
    index = 0
    
    output = widgets.Output()
    display(output)
    
    rows = []
    for root, dirs, files in os.walk(str(image_folder)):
        for file in files:
            if file.endswith(image_extension):
                abs_path = Path(root)/file
                rel_path = abs_path.relative_to(image_folder)
                rows.append([rel_path] + ['UNLABELED' for _ in attributes])
                index+=1
                if index%1000==0:
                    with output:
                        clear_output()
                        print(str(index))
    df = pandas.DataFrame(rows, columns=columns, dtype=pandas.CategoricalDtype)
    df.to_csv(csv_path, index_label='id')
    return df, rows, columns



from fastai.data_block import CategoryListBase
class LabelCls(CategoryListBase):
    
    def __init__(self, labels, path, **kwargs):
        # labels : array of class values e.g. [5,0,2]
        ATTRIBUTES = kwargs['ATTRIBUTES']
        kwargs.pop('ATTRIBUTES')
        
        self.one_hot_map = {}
        self.class_map = {}
        self.classes = []
        self.attribute_label_endpoints = []
        self.labels=[]
        global_class_idx = 0
        for attribute_idx, attribute in enumerate(ATTRIBUTES.keys()):
            label_segment_start = global_class_idx
            self.one_hot_map[attribute_idx]={}
            self.class_map[attribute_idx]={}
            classes_of_attribute = ATTRIBUTES[attribute][:-2]
            self.classes+=classes_of_attribute
            for class_idx, class_name in enumerate(classes_of_attribute):
                self.one_hot_map[attribute_idx][class_name]=global_class_idx
                self.class_map[attribute_idx][class_name]=class_idx
                global_class_idx+=1
            label_segment_end = global_class_idx
            self.attribute_label_endpoints.append((label_segment_start, label_segment_end))
        for label in labels:
            int_label = np.zeros(len(ATTRIBUTES), dtype=int)
            one_hot_label = np.zeros(len(self.classes), dtype=int)
            if isinstance(label, list):
                for attribute_idx, class_name in enumerate(label):
                    class_idx = self.one_hot_map[attribute_idx][class_name]
                    one_hot_label[class_idx]=1
                    int_label[attribute_idx]=self.class_map[attribute_idx][class_name]
            else:
                class_name=label
                class_idx = self.one_hot_map[0][class_name]
                one_hot_label[class_idx]=1
                int_label=self.class_map[0][class_name]
            
            self.labels.append(int_label)
        super().__init__(self.labels, classes=self.classes, **kwargs)
        
    def get(self, i):
        return self.labels[i]
