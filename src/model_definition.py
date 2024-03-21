import torch.nn as nn


__all__ = ['ANN_BinaryClassifier', 'CNN_LSTM_BinaryClassifier', 'CNN_BinaryClassifier']

class ANN_BinaryClassifier(nn.Module):
    '''
    A class representing a binary classification artificial neural network (ANN).

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output neurons (typically 1 for binary classification).

    Attributes:
        model (nn.Sequential): The sequential neural network model architecture.

    Methods:
        forward(x): Defines the forward pass of the neural network.

    '''

    def __init__(self, input_size, hidden_size, output_size = 1):
        '''
        Initializes the ANN_BinaryClassifier class.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output neurons (typically 1 for binary classification).
        '''

        super(ANN_BinaryClassifier, self).__init__()

        # Define the architecture of the neural network
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # Fully connected layer from input to hidden layer
            nn.ReLU(),  # ReLU activation function
            nn.Linear(hidden_size, output_size),  # Fully connected layer from hidden to output layer
            nn.Sigmoid()  # Sigmoid activation function for binary classification
        )

    def forward(self, x):
        '''
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the neural network.
        '''

        # Define the forward pass of the neural network
        return self.model(x)
    

class CNN_LSTM_BinaryClassifier(nn.Module):
    def __init__(self, input_channels=1, kernels=16, kernel_size=3, lstm_hidden_size=8, lstm_num_layers=1, classes=1):
        '''
        Convolutional Neural Network (CNN) followed by a Long Short-Term Memory (LSTM) for binary classification.

        Args:
            input_channels (int): Number of input channels.
            kernels (int): Number of kernels/filters in the convolutional layer.
            kernel_size (int): Size of the convolutional kernel.
            lstm_hidden_size (int): Number of features in the hidden state of the LSTM.
            lstm_num_layers (int): Number of recurrent layers in the LSTM.
            classes (int): Number of output classes (default is 1 for binary classification).
        '''
        super(CNN_LSTM_BinaryClassifier, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_channels, out_channels=kernels, kernel_size=kernel_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=kernels, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        '''
        Forward pass of the CNN-LSTM model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        '''
        x = self.cnn(x)
        x = self.tanh(x)
        out, _ = self.lstm(x)
        # Select the last output from the LSTM sequence
        out = self.fc(out[:, -1, :])
        # Apply the sigmoid activation function
        out = self.sigmoid(out)

        return out


class CNN_BinaryClassifier(nn.Module):
    def __init__(self, input_channels=1, kernels=16, kernel_size=3, input_features=10, classes=1):
        """
        Convolutional Neural Network (CNN) for binary classification.

        Args:
            input_channels (int): Number of input channels.
            kernels (int): Number of kernels/filters in the convolutional layer.
            kernel_size (int): Size of the convolutional kernel.
            input_features (int): Number of input features.
            classes (int): Number of output classes (default is 1 for binary classification).
        """
        super(CNN_BinaryClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=kernels, kernel_size=kernel_size),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(kernels * (input_features - kernel_size + 1), classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)

