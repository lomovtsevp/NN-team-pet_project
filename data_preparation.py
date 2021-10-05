import os
from pathlib import Path
from sklearn import preprocessing

import pandas as pd
import numpy as np

import torch
from torchvision import transforms
from PIL import Image

# Путь к train и test
PATH_TO_TRAIN = Path(r'C:\Users\User\PycharmProjects\pythonProject2\dataset\dataset\Train')
PATH_TO_TEST = Path(r'C:\Users\User\PycharmProjects\pythonProject2\dataset\dataset\Test')


def get_labels(file_names: list, label_enc: bool = False) -> list:
	"""
	:param file_names: list, полученный с помощью os.listdir()
	:param label_enc: bool, отвечает за использование LabelEncoder
	:return: list
	"""

	name_jpg = [i for i in file_names if i.find(".xml") == -1]
	labels = [name[:name.rfind("_")] for name in name_jpg]

	if label_enc:
		df = pd.DataFrame(labels)
		le = preprocessing.LabelEncoder()
		df[0] = le.fit_transform(df[0])
		labels = df[0].to_list()
	return labels


def create_paths_df(path) -> pd.DataFrame:
	"""
	Создает датафрейм с путями к картинкам
	:param path: путь к папке с данными
	:return: DataFrame с двумя столбцами:	img_dir - путь к картинке
											label - символ на картинке
	"""

	files = os.listdir(path)
	img_dir = list(path.glob(r'**/*.jpg'))
	label = get_labels(files, label_enc=True)

	# Create the paths data
	paths_df = pd.DataFrame(
		{
			'img_dir': img_dir,
			'label': label
		}).astype('str')
	return paths_df


def get_symbol_df(path_df: pd.DataFrame) -> pd.DataFrame:
	"""
	:param path_df: pd.DataFrame, датафрейм с путями к картинкам, полученный с помощью create_paths_df()
	:return: DataFrame, содержащий картинки перекодированные в числа.
	"""
	convert_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	len_df = path_df.shape[0]
	arr_of_tensors = []

	for i in range(len_df):
		img = Image.open(path_df.img_dir[i])
		tensor_of_img = torch.reshape(convert_tensor(img), (-1,))
		arr_of_tensors.append(np.array(tensor_of_img))

	symbol_df = pd.DataFrame(arr_of_tensors)
	symbol_df['target'] = path_df.label

	return symbol_df


def zip_df(path, name_out_file="out"):
	"""
	Создает датафрейм из картинок переведенных в тензоры и затем упаковывает данный датафрейм в архив.
	:param path: путь к Train или Test
	:param name_out_file: название архива и файла csv внутри
	:return: zip архив
	"""

	# Создаем датафреймы с путями к картинкам
	df = create_paths_df(path)

	# Перемешиваем их
	df = df.sample(frac=1).reset_index(drop=True)

	# Получаем тензоры из картинок, а затем датафрейм из тензора и лейбла.
	df = get_symbol_df(df)

	# сохраняем полученный датафрейм в архив с csv файлом
	compression_opts = dict(method='zip', archive_name=f'{name_out_file}.csv')
	df.to_csv(f'{name_out_file}.zip', compression=compression_opts)


if __name__ == "__main__":
	zip_df(PATH_TO_TEST, "test_df")
	zip_df(PATH_TO_TRAIN, "train_df")
