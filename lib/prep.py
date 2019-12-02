import os
import pandas
from pathlib import Path
from ipywidgets import widgets, Layout
from IPython.display import clear_output, display


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