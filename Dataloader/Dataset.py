import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pathlib import Path
import zipfile

# Take a look at the list dataset
def list_of_dataset():
  list_dataset = [
    "Bag_Classes",
    "Bag4Classes",
    "Data_real",
    "dataset_capstone_9",
    "Drinking_waste_classification",
    "Garbage_classification_2",
    "garbage_classification_3",
    "Garbage_classification_dataset",
    "Garbage_classification_enhanced",
    "garbage_dataset",
    "garbage1_dataset",
    "trash_dataset",
    "trashify-image-dataset",
    "TrashType_Image_Dataset",
    "waste_dataset"
  ]
  print(list_dataset)
  return list_dataset

# Download and extract the dataset
def download_and_extract_zips(selected_datasets, download_dir='/content/Dataset',extract_dir='/content/data'):
    os.makedirs(download_dir, exist_ok=True)
    
    extracted_paths = []
    
    for dataset in selected_datasets:
        
        url = "http://clouds.iec-uit.com/smartbin.dataloader/"
        full_url = url + dataset + '.zip'
        
       
        zip_path = os.path.join(download_dir, dataset + '.zip')
        
      
        archive = tf.keras.utils.get_file(fname=zip_path, origin=full_url, extract=False)
        if extract_dir:
            extract_path = os.path.join(extract_dir, dataset)
        else:
            extract_path = os.path.splitext(archive)[0]
        
        os.makedirs(extract_path, exist_ok=True)
        
      
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        inner_folder = os.path.join(extract_path, dataset)
        if os.path.isdir(inner_folder):
            for item in os.listdir(inner_folder):
                os.rename(os.path.join(inner_folder, item), os.path.join(extract_path, item))
            os.rmdir(inner_folder)
    
        extracted_paths.append(Path(extract_path))
    
  
    for path in extracted_paths:
        print(f"Dataset extracted to: {path}")
    
    return extracted_paths

def load_dataset(dataset_path,Target_size =(224, 224),Color_mode = 'rgb',Class_mode = 'categorical',Batch_size = 32, Shuffle = True,Seed = 42):

  data_dir = Path(dataset_path)

  # Get filepaths and labels
  filepaths = list(data_dir.glob(r'**/*.JPG')) + list(data_dir.glob(r'**/*.jpg')) + list(data_dir.glob(r'**/*.png')) + list(data_dir.glob(r'**/*.png'))

  labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

  filepaths = pd.Series(filepaths, name='Filepath').astype(str)
  labels = pd.Series(labels, name='Label')

  # Concatenate filepaths and labels
  image_df = pd.concat([filepaths, labels], axis=1)

  train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input,
    validation_split=0.2
  )

  test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.resnet.preprocess_input
  )

  train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

  # Split the data into three categories.
  train_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=Target_size,
    color_mode=Color_mode,
    class_mode=Class_mode,
    batch_size=Batch_size,
    shuffle=Shuffle,
    seed=Seed,
    subset='training'
  )

  val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=Target_size,
    color_mode=Color_mode,
    class_mode=Class_mode,
    batch_size=Batch_size,
    shuffle=Shuffle,
    seed=Seed,
    subset='validation'
  )

  test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=Target_size,
    color_mode=Color_mode,
    class_mode=Class_mode,
    batch_size=Batch_size,
    shuffle=Shuffle,
    seed=Seed,
  )
  return train_images, val_images, test_images

def Data_visualization(generator):
# Lấy toàn bộ dữ liệu và tính số lượng mẫu theo nhãn
  num_images = len(generator.filenames)
  labels = np.zeros((num_images,))
  for i in range(num_images // generator.batch_size + 1):
    batch_images, batch_labels = next(generator)
    start_idx = i * generator.batch_size
    end_idx = start_idx + batch_images.shape[0]
    labels[start_idx:end_idx] = np.argmax(batch_labels, axis=1)  # Lấy chỉ số lớp có giá trị lớn nhất

  # Đếm số lượng mẫu cho từng nhãn
  unique_labels, label_counts = np.unique(labels, return_counts=True)

  # Trực quan hóa dữ liệu
  plt.figure(figsize=(10, 6))
  plt.bar(unique_labels, label_counts, color='skyblue')
  plt.xlabel('label')
  plt.ylabel('Sample')
  plt.title('Data visualization')
  plt.xticks(unique_labels)
  plt.show()

  