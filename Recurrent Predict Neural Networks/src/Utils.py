import math
import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from scipy.io import arff
from sklearn.model_selection import train_test_split


def get_dataset_name(modelConfig):
    return modelConfig["select_dataset"]

def get_filepath(modelConfig):
    return modelConfig["data_filepath"] + modelConfig["dataset"][modelConfig["select_dataset"]][0]

def get_class_index(modelConfig):
    return modelConfig["dataset"][modelConfig["select_dataset"]][1]

def get_save_filepath(modelConfig):
    return modelConfig["save_filepath"] + modelConfig["select_dataset"] + "/" + modelConfig["select_dataset"]


class Load_Dataset:
    def load(self,modelConfig):
        filepath = get_filepath(modelConfig)
        data = load_dataset_to_dataframe(filepath)

        class_index = get_class_index(modelConfig)
        if class_index is None or class_index == -1:
            class_index = data.shape[-1]-1

        data_index = list(range(data.shape[-1]))
        data_index.remove(class_index)
        print(data.shape[-1])
        print(data_index)

        # print(data_index)

        convert(data)

        print(data.head())
        data = data.to_numpy()
        X = data[:,data_index]
        y = data[:,class_index]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)
        device = torch.device(modelConfig["device"])

        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float64).to(device)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
        self.X_test_tensor = torch.tensor(X_test, dtype=torch.float64).to(device)
        self.y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

        print("X_train_tensor:",self.X_train_tensor.shape)
        print("y_train_tensor:",self.y_train_tensor.shape)
        print("X_test_tensor:",self.X_test_tensor.shape)
        print("y_test_tensor:",self.y_test_tensor.shape)
        self.feature = len(self.X_train_tensor[0])
        self.n_clusters = len(np.unique(data[:,class_index]))
        print("class:", self.n_clusters)


def exchange(tensor1,tensor2):
    return tensor2,tensor1


def load_dataset_to_dataframe(filepath):
    file_extension = filepath.split('.')[-1].lower()

    if file_extension == 'csv':
        df = pd.read_csv(filepath, delimiter=';', skiprows=1, header=None)
    elif file_extension == 'xlsx':
        df = pd.read_excel(filepath)
    elif file_extension == 'xls':
        df = pd.read_excel(filepath)
    elif file_extension in ['data', 'txt']:
        df = pd.read_csv(filepath, header=None)
    elif file_extension == 'arff':
        data, meta = arff.loadarff(filepath)
        df = pd.DataFrame(data)
        df['Class'] = df['Class'].apply(lambda x: x.decode('utf-8'))
        print(df.head())
    else:
        raise ValueError(f"不支持的文件类型: {file_extension}")

    return df


def convert(data):
    is_string = pd.Series(dtype=bool)

    for column in data:
        is_string[column] = data[column].apply(lambda x: isinstance(x, str)).any()

    for column_name, is_str in is_string.items():
        if is_str:
            unique_categories = data[column_name].unique()

            category_map = {category: code for code, category in enumerate(unique_categories)}

            data[column_name] = data[column_name].map(category_map)



def truncate_or_pad_list(lst, target_length, fill_value=None):
    return lst[:target_length] + [fill_value] * (target_length - len(lst))


def add_offset_to_list(data_list, offset):
    return [value + offset if value is not None else None for value in data_list]


def save_show(accuracys,modelConfig):
    data = {
        'step' : list(range(1,modelConfig["step_epochs"]+1)),
        'accuracys': accuracys
    }

    dataset_name = get_dataset_name(modelConfig)
    save_filepath = get_save_filepath(modelConfig)
    if not os.path.exists(os.path.dirname(save_filepath)):
        os.makedirs(os.path.dirname(save_filepath))

    df = pd.DataFrame(data)
    df.to_csv(f"{save_filepath}.csv", index=False)


    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(list(range(1,modelConfig["step_epochs"]+1)), accuracys, color="tab:blue", linestyle='-', linewidth=2,label="Neural Networks Accuracy")

    ax.set_xlabel("step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Repeat neural networks on the {dataset_name}")

    ax.grid(visible=True, linestyle='--', alpha=0.5)

    plt.legend(loc='lower right', fontsize=12, ncol=3)
    ax.tick_params(axis='both', labelsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    plt.savefig(f"{save_filepath}_accuracy.jpg",dpi=600)
    plt.show()