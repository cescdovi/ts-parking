from torch.utils.data import Dataset
import torch

class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_window, output_window, feature_cols, target_col):
        """
        data: DataFrame que contiene los datos de la serie temporal
        input_window: nº de pasos de tiempo en la secuencia de entrada
        output_window: nº de pasos de tiempo a predecir
        feature_cols: lista de nombres de columnas que se usan como característcas
        target_col: nombre de la variable a predecir
        """
        self.data = data
        self.input_window = input_window
        self.output_window = output_window
        self.feature_cols = feature_cols
        self.target_cols = target_col

    def __len__(self):
        """
        Función que devuele el nº de datos del Dataset
        """
        return len(self.data) - self.input_window - self.output_window + 1 #

    def __getitem__(self, idx):
        """
        Función que devuelve un dato a partir de un índice
        """
        X = self.data[idx: idx + self.input_window][self.feature_cols].values
        Y = self.data[idx + self.input_window: idx + self.input_window + self.output_window][self.target_cols].values
        
        X_tensor = torch.tensor(X, dtype= torch.float32)
        Y_tensor = torch.tensor(Y, dtype= torch.float32)

        return X_tensor, Y_tensor