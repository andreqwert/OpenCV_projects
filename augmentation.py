import os
import cv2
import numpy as np
from scipy import ndimage


class Process:
	def __init__(self, image_path, what_to_do=None, *args, **kwargs):
		self._img = cv2.imread(image_path, 1)
		self.h = self._img.shape[0]
		self.w = self._img.shape[1]
		self.center = (self.w/2, self.h/2)
		if what_to_do:
			self.__getattribute__(what_to_do)(*args, **kwargs)

	def rotate(self, angle=90, scale=1.0):
		"""Повернуть изображение"""

		rotated = ndimage.rotate(self._img, angle)
		cv2.imwrite('rotated.png', rotated)
		return rotated

	def bias_x(self, offset_x=20):
		"""Смещение изображения по оси X"""

		rows, cols, _ = self._img.shape
		M = np.float32([[1, 0, offset_x], [0, 1, 0]])
		biased_x = cv2.warpAffine(self._img, M, (cols, rows))
		cv2.imwrite('biased_x.png', biased_x)
		return biased_x

	def bias_y(self, offset_y=50):
		"""Смещение изображения по оси Y"""

		rows, cols, _ = self._img.shape
		M = np.float32([[1, 0, 0], [0, 1, offset_y]])
		biased_y = cv2.warpAffine(self._img, M, (cols, rows))
		cv2.imwrite('biased_y.png', biased_y)
		return biased_y

	def compress(self):
		"""Сжать изображение (ухудшить качество <-> понизить занимаемое место на диске)"""

		cv2.imwrite('compressed.png', self._img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
		return None

	def resize(self, width=440, height=200):
		"""Растянуть/cжать изображение (изменить размер)"""

		dim = (width, height)
		resized = cv2.resize(self._img, dim, interpolation=cv2.INTER_AREA)
		cv2.imwrite('resized.png', resized)
		return resized

	def translate(self, offset=100):
		"""Параллельный перенос изображения"""

		rows, cols, _ = self._img.shape
		M = np.float32([[1, 0, offset], [0, 1, offset]])
		translated = cv2.warpAffine(self._img, M, (cols, rows))
		cv2.imwrite('translated.png', translated)
		return translated

	def flip(self):
		"""Симметрия изображения"""

		flipped_image = cv2.flip(self._img,1)
		cv2.imwrite('flipped.png', flipped_image)
		return flipped_image







Process(os.getcwd() + '/1.jpg')
Process(os.getcwd() + '/1.jpg', 'rotate')
Process(os.getcwd() + '/1.jpg', 'bias_x')
Process(os.getcwd() + '/1.jpg', 'bias_y')
Process(os.getcwd() + '/1.jpg', 'compress')
Process(os.getcwd() + '/1.jpg', 'resize')
Process(os.getcwd() + '/1.jpg', 'translate')
Process(os.getcwd() + '/1.jpg', 'flip')


