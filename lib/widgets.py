from ipywidgets import widgets, Layout
from IPython.display import clear_output, display
import pandas
from functools import partial
from fastprogress import progress_bar

class MultilabelerWidget:
    def __init__(self, csv_path, image_folder, image_column, attributes, width=300, height=300):
        self.image_folder = image_folder
        self.image_column = image_column
        self.attributes = attributes
        self.csv_path = csv_path
        self.original_df = pandas.read_csv(csv_path)
        self.df = self.original_df
        self.max = len(self.df)
        self.width = width
        self.height = height
        self.size = 2
        self.rows = None
        self.to_beginning()
        self.last_rows = self.rows
        self.attribute_dropdown_mappings = {}
        self.render()
    
    def to_beginning(self):
        if self.rows is not None:
            self.last_rows = self.rows
        self.rows = list(range(self.size))
        self.last_index = self.rows[-1]
        
    def on_attribute_value_change(self, change):
        original_row_index, attribute_name = self.attribute_dropdown_mappings[change.owner]
        new_value = change.new
        self.original_df.at[original_row_index, attribute_name] = new_value
        self.original_df.to_csv(self.csv_path, index_label='id')
        
    def next_batch(self, event):
        self.last_rows = self.rows
        self.rows = [(i+self.last_index+1)%self.max for i in range(self.size)]
        self.last_index = self.rows[-1]
        self.render()
        
    def prev_batch(self, event):
        temp = self.rows
        self.rows = self.last_rows
        self.last_rows = temp
        self.last_index = self.rows[-1]
        self.children = self.generate_row_widgets(self.rows)
        self.container = widgets.HBox(self.children)
        self.render()
        
    def generate_row_widgets(self, rows=[0]):
        child_widgets = []
        for index in rows:
            row = self.df.iloc[index]
            row_index = row['id']
            child_components = []
            
            #image
            file_path = row[self.image_column]
            with open(self.image_folder/file_path, "rb") as file:
                image = file.read()
                image_widget = widgets.Image(
                    value = image,
                    format = 'jpg',
                    width = self.width,
                    height = self.height
                )
                child_components.append(image_widget)
                
            #attribute dropdowns
            for key in self.attributes.keys():
                attribute_name = key
                attribute_options = self.attributes[key]
                dropdown_widget = widgets.Dropdown(
                    options = attribute_options,
                    description = attribute_name,
                    disabled = False,
                    value = row[attribute_name]
                )
                self.attribute_dropdown_mappings[dropdown_widget]=(row_index, attribute_name)
                dropdown_widget.observe(self.on_attribute_value_change, names='value')
                child_components.append(dropdown_widget)
            
            
            #pack into containers
            child_container = widgets.VBox(child_components, width = 50)
            child_widgets.append(child_container)
            
        #buttons and stuff
        control_components = self.generate_control_widgets()
        control_container = widgets.VBox(control_components, width = 50)
        child_widgets.append(control_container)
            
        return child_widgets
    
    def generate_control_widgets(self):
        control_components = []
        next_button = widgets.Button(description="Next")
        next_button.on_click(self.next_batch)
        control_components.append(next_button)
        prev_button = widgets.Button(description="Back")
        prev_button.on_click(self.prev_batch)
        control_components.append(prev_button)
        return control_components

    def render(self):
        
        self.children = self.generate_row_widgets(self.rows)
        self.container = widgets.HBox(self.children)
        clear_output()
        display(self.container)
        

class MultilabelerActiveLearningWidget(MultilabelerWidget):
    
    def __init__(self, learner, unlabeled_tag, classifier_export, csv_path, image_folder, image_column, attributes, width=300, height=300):
        
        self.attribute = list(attributes)[0]
        self.unlabeled_tag = unlabeled_tag
        super().__init__(csv_path, image_folder, image_column, attributes, width, height)
        self.original_df = self.df
        self.learner = learner
        assert(self.learner is not None)
        self.unlabeled_list = None
        self.df = None
        self.sort_by_confusion()
        self.render()
        
    def train_epochs(self, event):
        self.learner.fit_one_cycle(1)
        
    def sort_by_confusion(self, event=None):
        import fastai
        from fastai.vision import ImageList
        import torch
        import torch.nn.functional as F
        self.df = self.original_df[self.original_df[self.attribute]==self.unlabeled_tag][:300]
        path_stupid_fix = self.image_folder.parent.parent
        unlabeled_il = ImageList.from_df(self.df,
                                           path=path_stupid_fix,
                                           folder=self.image_folder.relative_to(path_stupid_fix), suffix=None, 
                                           cols=self.image_column)

        probs = torch.zeros([len(unlabeled_il), self.learner.data.c])  # todo 2 -> num classes
        with torch.no_grad():
            for idx in progress_bar(range(len(unlabeled_il))):
                instance = self.learner.data.one_item(unlabeled_il[idx])
                out = self.learner.model(instance[0])
                prob = F.softmax(out, dim=1)
                probs[idx] += prob.cpu()[0]
            probs /=2

        log_probs = torch.log(probs) # most confusing will be low
        U = (probs * log_probs).sum(1) # most confusing will still be low
        requests = U.sort()[1] # sorts in ascending order - so most confusing first
        print("SORT DF", len(self.df))
        print("SORT REQUESTS", len(requests))
        self.df.reindex(requests)
        self.to_beginning()
        self.render()
        
    def on_attribute_change(self, change):
        new_value = change.new
        self.attribute = new_value
        self.sort_by_confusion()
        
    def generate_control_widgets(self):
        control_components = []
        
        attribute_dropdown = dropdown_widget = widgets.Dropdown(
            options = self.attributes.keys(),
            description = 'Attribute',
            disabled = False,
            value = self.attribute
        )
        attribute_dropdown.observe(self.on_attribute_change, names='value')
        control_components.append(attribute_dropdown)
        
        next_button = widgets.Button(description="Next")
        next_button.on_click(self.next_batch)
        control_components.append(next_button)
        prev_button = widgets.Button(description="Back")
        prev_button.on_click(self.prev_batch)
        control_components.append(prev_button)
        train_button = widgets.Button(description="Train")
        train_button.on_click(self.train_epochs)
        control_components.append(train_button)
        sort_button = widgets.Button(description="Sort")
        sort_button.on_click(self.sort_by_confusion)
        control_components.append(sort_button)
        
        return control_components
        
