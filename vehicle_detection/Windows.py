import numpy as np

class Windows():
	def __init__(self, n):
		self.n = n
		self.windows = []

	def add_windows(self, new_windows):
		self.windows.append(new_windows)

		q_full = len(self.windows) >= self.n
		if q_full:
			_ = self.windows.pop(0)

	def get_windows(self):
		out_windows = []
		for window in self.windows:
			out_windows = out_windows + window
		return out_windows
