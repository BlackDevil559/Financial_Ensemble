import math
import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, filename, split, cols, cols_to_norm, pred_len):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.cols_to_norm = cols_to_norm
        self.pred_len = pred_len
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise, cols_to_norm):
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)
        y_base = data_windows[:, 0, [0]]
        data_windows = self.normalise_selected_columns(data_windows, cols_to_norm, single_window=False) if normalise else data_windows
        cut_point = self.pred_len
        x = data_windows[:, :-1, :]
        y = data_windows[:, -1, [0]]
        return x,y,y_base

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len,normalise):
        window = self.data_train[i:i+seq_len]
        window = self.normalise_selected_columns(window, self.cols_to_norm, single_window=True)[0] if normalise else window       
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
# 
    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                w = window[0, col_i]
                if w == 0:
                  w = 1
                normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T 
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    def normalise_selected_columns(self, window_data, columns_to_normalise, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                if col_i in columns_to_normalise:
                    
                    w = window[0, col_i]
                    if w == 0:
                        w = 1
                    normalised_col = [((float(p) / float(w)) - 1) for p in window[:, col_i]]
                else:
                    normalised_col = window[:, col_i].tolist()
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
