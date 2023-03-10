import array
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device}" " is available.")


# https://colab.research.google.com/drive/1enI68fTdPI2w5KKv6jyL0Lcq9Zg3BbLx?usp=sharing#scrollTo=gMO3hUZfPA8y
# Adaptado aos dados da Dissertação

import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.style.use('seaborn-darkgrid')
plt.rcParams["figure.figsize"] = (7, 5)

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, LSTM
from keras.metrics import RootMeanSquaredError
import datetime

import warnings
warnings.filterwarnings("ignore")
import os
os.getcwd()
from deap import algorithms, base, creator, tools
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates an RNN instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(RNNModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim

        # RNN layers
        self.rnn = nn.RNN(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, h0 = self.rnn(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)
        return out

class LSTMModel(nn.Module):
    """LSTMModel class extends nn.Module class and works as a constructor for LSTMs.

       LSTMModel class initiates a LSTM module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           lstm (nn.LSTM): The LSTM model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of LSTMs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, func_act, op, seed):
        """The __init__ method that initiates a LSTM instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(LSTMModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.seed = seed
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )
        self.func_act = func_act
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        random.seed(self.seed)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = self.func_act(out)
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

class GRUModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constructor for GRUs.

       GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(GRUModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        # GRU layers
        self.gru = nn.GRU(
            input_dim, hidden_dim, layer_dim, batch_first=True, dropout=dropout_prob
        )

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out

class MLPModel(nn.Module):
    """ Texto sobre MLP

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           linear (nn.Linear): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, func_act, op, seed):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(MLPModel, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_prob)
        self.seed = seed
        # MLP layers
        self.mlp = nn.Linear(
            input_dim, input_dim #, layer_dim
        )
        self.func_act = func_act
        # Fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # # Initializing hidden state for first input with zeros
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        #
        # # Forward propagation by passing in the input and hidden state into the model
        # #out = self.mlp(x, h0.detach())
        # out = self.mlp(x, x.size(0))
        # print(out)
        # # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # # so that it can fit into the fully connected layer
        # out = out[:, -1, :]
        # #out = out[:, -1]
        #
        # # Convert the final state to our desired output shape (batch_size, output_dim)
        # out = self.fc(out)
        random.seed(self.seed)
        x = self.mlp(x)
        x = self.dropout(x)
        out = self.func_act(x)
        out = self.fc(out)
        return out

class CONV1DModel(nn.Module):
    """GRUModel class extends nn.Module class and works as a constructor for GRUs.

       GRUModel class initiates a GRU module based on PyTorch's nn.Module class.
       It has only two methods, namely init() and forward(). While the init()
       method initiates the model with the given input parameters, the forward()
       method defines how the forward propagation needs to be calculated.
       Since PyTorch automatically defines back propagation, there is no need
       to define back propagation method.

       Attributes:
           hidden_dim (int): The number of nodes in each layer
           layer_dim (str): The number of layers in the network
           gru (nn.GRU): The GRU model constructed with the input parameters.
           fc (nn.Linear): The fully connected layer to convert the final state
                           of GRUs to our desired output shape.

    """
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, dropout_prob, func_act, op, seed):
        """The __init__ method that initiates a GRU instance.

        Args:
            input_dim (int): The number of nodes in the input layer
            hidden_dim (int): The number of nodes in each layer
            layer_dim (int): The number of layers in the network
            output_dim (int): The number of nodes in the output layer
            dropout_prob (float): The probability of nodes being dropped out

        """
        super(CONV1DModel, self).__init__()

        # # Defining the number of layers and the nodes in each layer
        # self.layer_dim = layer_dim
        # self.hidden_dim = hidden_dim
        #
        # # GRU layers
        # self.conv1d = nn.Conv1d(
        #     1, 64, output_dim, stride=2
        # )
        #
        # # Fully connected layer
        # self.fc = nn.Linear(output_dim * 6, 64 ** 2)
        self.seed = seed
        self.conv1d = nn.Conv1d(1,64,kernel_size=1)
        self.func_act = func_act
        # self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(11, 64**2)
        self.fc2 = nn.Linear(64**2, output_dim)

    def forward(self, x):
        """The forward method takes input tensor x and does forward propagation

        Args:
            x (torch.Tensor): The input tensor of the shape (batch size, sequence length, input_dim)

        Returns:
            torch.Tensor: The output tensor of the shape (batch size, output_dim)

        """
        # # Initializing hidden state for first input with zeros
        # h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        #
        # # Forward propagation by passing in the input and hidden state into the model
        # out, _ = self.conv1d(x, h0.detach())
        # print(out)
        #
        # # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # # so that it can fit into the fully connected layer
        # out = out[:, -1, :]
        #
        # # Convert the final state to our desired output shape (batch_size, output_dim)
        # out = self.fc(out)
        # out = self.conv1d(x)
        # print(out)
        # out = self.fc(out)
        # print(out)
        random.seed(self.seed)
        out = self.conv1d(x)
        out = self.func_act(out)
        # out = out[:, -1, :]
        # out = self.relu(out)
        # out = out.view(-1)
        out = self.fc1(out)
        # out = self.relu(out)
        out = self.fc2(out)
        out = out[:, -1, :]
        print(out)

        return out

class Optimization:
    """Optimization is a helper class that allows training, validation, prediction.

    Optimization is a helper class that takes model, loss function, optimizer function
    learning scheduler (optional), early stopping (optional) as inputs. In return, it
    provides a framework to train and validate the models, and to predict future values
    based on the models.

    Attributes:
        model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
        loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
        optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        train_losses (list[float]): The loss values from the training
        val_losses (list[float]): The loss values from the validation
        last_epoch (int): The number of epochs that the models is trained
    """
    def __init__(self, model, loss_fn, optimizer):
        """
        Args:
            model (RNNModel, LSTMModel, GRUModel): Model class created for the type of RNN
            loss_fn (torch.nn.modules.Loss): Loss function to calculate the losses
            optimizer (torch.optim.Optimizer): Optimizer function to optimize the loss function
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        """The method train_step completes one step of training.

        Given the features (x) and the target values (y) tensors, the method completes
        one step of the training. First, it activates the train mode to enable back prop.
        After generating predicted values (yhat) by doing forward propagation, it calculates
        the losses by using the loss function. Then, it computes the gradients by doing
        back propagation and updates the weights by calling step() function.

        Args:
            x (torch.Tensor): Tensor for features to train one step
            y (torch.Tensor): Tensor for target values to calculate losses

        """
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=64, n_epochs=50, n_features=1):
        """The method train performs the model training

        The method takes DataLoaders for training and validation datasets, batch size for
        mini-batch training, number of epochs to train, and number of features as inputs.
        Then, it carries out the training by iteratively calling the method train_step for
        n_epochs times. If early stopping is enabled, then it  checks the stopping condition
        to decide whether the training needs to halt before n_epochs steps. Finally, it saves
        the model in a designated file path.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader that stores training data
            val_loader (torch.utils.data.DataLoader): DataLoader that stores validation data
            batch_size (int): Batch size for mini-batch training
            n_epochs (int): Number of epochs, i.e., train steps, to train
            n_features (int): Number of feature columns

        """
        model_path = f'{self.model}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            # if (epoch <= 10) | (epoch % 50 == 0):
            #     print(
            #         f"[{epoch}/{n_epochs}] Training loss: {training_loss:.4f}\t Validation loss: {validation_loss:.4f}"
            #     )

        torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size=1, n_features=1):
        """The method evaluate performs the model evaluation

        The method takes DataLoaders for the test dataset, batch size for mini-batch testing,
        and number of features as inputs. Similar to the model validation, it iteratively
        predicts the target values and calculates losses. Then, it returns two lists that
        hold the predictions and the actual values.

        Note:
            This method assumes that the prediction from the previous step is available at
            the time of the prediction, and only does one-step prediction into the future.

        Args:
            test_loader (torch.utils.data.DataLoader): DataLoader that stores test data
            batch_size (int): Batch size for mini-batch training
            n_features (int): Number of feature columns

        Returns:
            list[float]: The values predicted by the model
            list[float]: The actual values in the test set.

        """
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().numpy())
                values.append(y_test.to(device).detach().numpy())

        return predictions, values

    def plot_losses(self):
        """The method plots the calculated loss values for training and validation
        """
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


class DEAP:
    ''' Exploracão de Arquitectura de redes neurais utilizando o Algoritmo evolucionario
     https://github.com/DEAP/deap '''
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.optimizers = ['SGD', 'RMSprop', 'Adam','Adadelta','Adamax','SGD','Adam','Adadelta','Adamax']
        #self.qtde_camada = [10, 20, 30, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
        self.tamanho_camada = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        # self.loss = ['mse','mae','map'] # Vou optar por loss = MSE

        # self.metricas = ['rmse','mse','mae'] # Vou setar a métrica do algoritmo igual a do artigo
        self.activation_keras = ['sigmoid', 'relu', 'selu','elu' ,'tanh',
                                 'softmax', 'softplus', 'softsign', 'exponential']
        # O DEAP precisa que o número de parametros seja igual ao número de itens ==> Se eu tenho 9 Funcoes de ativacao em Keras, preciso de 9 parametros.
        # Ainda nao encontrei uma soluçao elegante, a melhor até agora foi incluir params vazios que nao serao usados na arquitetura.
        self.param_vazio_1 = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.50]
        self.param_vazio_2 = [x for x in range(10,45,4)]
        self.param_vazio_3 = ['mean_squared_error','cosine_similarity','mean_absolute_error',
                              'mean_absolute_percentage_error','mean_squared_logarithmic_error','cosine_similarity',
                              'logcosh','huber_loss','mean_absolute_error',]
        self.param_vazio_4 = list(np.random.randint(10, 100, 9))
        self.param_vazio_5 = [x for x in range(0,9,1)]
        self.param_vazio_6 = [x for x in range(0,9,1)]
        # Exclusivos Para o modelo MLP Regressor sklearn
        self.hidden_layer = [2, 4, 8, 16, 32, 64, 128]
        self.alpha = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        #self.activation = ['identity', 'logistic', 'relu', 'tanh', 'relu','tanh'] #'softmax'
        self.activation = [nn.Tanh(), nn.Sigmoid(), nn.Softmax(),
                           nn.ReLU(), nn.ELU(), nn.Softplus(), nn.SELU() ] #'softmax'
        #self.batch_size = [50, 100, 150, 200, 250, 300, 350]
        self.batch_size = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        self.learning_rate = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
        #self.max_iter = [50, 100, 150, 200, 250, 300]
        #self.max_iter = [optim.ASGD, optim.LBFGS, optim.NAdam, optim.RAdam, optim.Adam, optim.RMSprop]
        self.max_iter = [optim.ASGD, optim.NAdam, optim.RAdam,
                         optim.Adam, optim.RMSprop, optim.Adamax, optim.SGD]
        # self.max_iter = [optim.NAdam, optim.NAdam, optim.NAdam,
        #                  optim.NAdam, optim.NAdam, optim.NAdam, optim.NAdam]
        self.fator_aleatorio = [9, 21, 42, 66, 72, 98, 64]


    def get_scaler(self, scaler):
        scalers = {
            "minmax": MinMaxScaler,
            "standard": StandardScaler,
            "maxabs": MaxAbsScaler,
            "robust": RobustScaler,
        }
        return scalers.get(scaler.lower())()

    def train_teste_base_torch(self, cv1d = False):
        batch_size = 64
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                shuffle=False)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                              test_size=0.2, shuffle=False)

        scaler = self.get_scaler('minmax')
        self.X_train_arr = scaler.fit_transform(self.X_train)
        self.X_val_arr = scaler.transform(self.X_val)
        self.X_test_arr = scaler.transform(self.X_test)
        self.X_arr = scaler.transform(self.X)
        self.y_train_arr = scaler.fit_transform(self.y_train.values.reshape(-1,1))
        self.y_val_arr = scaler.transform(self.y_val.values.reshape(-1,1))
        self.y_test_arr = scaler.transform(self.y_test.values.reshape(-1,1))
        self.y_arr = scaler.transform(self.y.values.reshape(-1,1))
        train_features = torch.Tensor(self.X_train_arr)
        train_targets = torch.Tensor(self.y_train_arr)
        val_features = torch.Tensor(self.X_val_arr)
        val_targets = torch.Tensor(self.y_val_arr)
        test_features = torch.Tensor(self.X_test_arr)
        test_targets = torch.Tensor(self.y_test_arr)
        full_features = torch.Tensor(self.X_arr)
        full_targets = torch.Tensor(self.y_arr)

        if cv1d:
            self.train = TensorDataset(train_features.reshape(train_features.shape[0],train_features.shape[1],1),
                                       train_targets)
            self.val = TensorDataset(val_features.reshape(val_features.shape[0],val_features.shape[1],1), val_targets)
            self.test = TensorDataset(test_features.reshape(test_features.shape[0],test_features.shape[1],1),
                                      test_targets)
        else:
            self.train = TensorDataset(train_features, train_targets)
            self.val = TensorDataset(val_features, val_targets)
            self.test = TensorDataset(test_features, test_targets)
            self.full = TensorDataset(full_features, full_targets)

        self.train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=False, drop_last=True)
        self.val_loader = DataLoader(self.val, batch_size=batch_size, shuffle=False, drop_last=True)
        self.test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=False, drop_last=True)
        self.test_loader_one = DataLoader(self.test, batch_size=1, shuffle=False, drop_last=True)
        self.full_loader_one = DataLoader(self.full, batch_size=1, shuffle=False, drop_last=True)
        return self

    def get_model(self, model, model_params):
            self.models = {
                "rnn": RNNModel,
                "lstm": LSTMModel,
                "gru": GRUModel,
                "mlp": MLPModel,
                "conv1d": CONV1DModel,
            }
            return self.models.get(model.lower())(**model_params)
            #return self

    def inverse_transform(self, scaler, df, columns):
        self.df = df
        for col in columns:
            self.df[col] = scaler.inverse_transform(df[col])
        # return df
        return self.df

    def format_predictions(self, predictions, values, df_test, scaler):
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        self.df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
        self.df_result = self.df_result.sort_index()
        self.df_result = self.inverse_transform(scaler, self.df_result, [["value", "prediction"]])
        # return df_result
        return self.df_result

    def eval_model_LSTM(self, params, full=False):
        '''
        Gera uma avaliação do individuo - utilizando uma RNN LSTM Pytorch
        :param params: Recebe os parametros do individuo no DEAP
        :return: RMSE do Individuo
        '''

        input_dim = len(self.X_train.columns)
        output_dim = 1
        self.params = params
        #hidden_dim = 64
        hidden_dim = self.hidden_layer[self.params[0]]
        #layer_dim = 3
        layer_dim = 3
        batch_size = 64
        #batch_size = self.hidden_layer[self.params[0]]
        #dropout = 0.2
        dropout = self.batch_size[self.params[3]]
        n_epochs = 20
        #learning_rate = 1e-3
        learning_rate = self.learning_rate[self.params[4]]
        #weight_decay = 1e-6
        weight_decay = self.alpha[self.params[2]]
        func_act = self.activation[self.params[1]]
        op = self.max_iter[self.params[5]]
        seed = self.fator_aleatorio[self.params[6]]
        model_params = {'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'layer_dim': layer_dim,
                        'output_dim': output_dim,
                        'dropout_prob': dropout,
                        'func_act': func_act,
                        'op': op,
                        'seed': seed}
        model = self.get_model('lstm', model_params)

        loss_fn = nn.MSELoss(reduction="mean")
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = op(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 0.548

        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(self.train_loader, self.val_loader, batch_size=batch_size,
                  n_epochs=n_epochs, n_features=input_dim)
        #opt.plot_losses()

        predictions, values = opt.evaluate(
            self.test_loader_one,
            batch_size=1,
            n_features=input_dim
        )
        scaler = self.get_scaler('minmax')
        scaler.fit(self.y_train.values.reshape(-1,1))
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        self.df_result = pd.DataFrame(data={"value": vals, "prediction": preds},
                                 index=self.X_test.head(len(vals)).index)
        self.df_result = self.df_result.sort_index()
        self.df_result = self.inverse_transform(scaler, self.df_result, [["value", "prediction"]])
        if full:
            inicio_predicao = datetime.datetime.now()
            predictions, values = opt.evaluate(
                self.full_loader_one,
                batch_size=1,
                n_features=input_dim
                )
            fim_predicao = datetime.datetime.now()
            print(f"Duracao da Previsao DEAP/LSTM: {fim_predicao - inicio_predicao}")
            scaler = self.get_scaler('minmax')
            scaler.fit(self.y.values.reshape(-1,1))
            vals = np.concatenate(values, axis=0).ravel()
            preds = np.concatenate(predictions, axis=0).ravel()
            self.df_result_full = pd.DataFrame(data={"value": vals, "prediction": preds},
                                          index=self.X.index)
            self.df_result_full = self.df_result_full.sort_index()
            self.df_result_full = self.inverse_transform(scaler, self.df_result_full, [["value", "prediction"]])
            return self
        #print(df_result)
        mse = (np.square(self.df_result.value.values - self.df_result.prediction.values)).mean(axis=None)
        self.rmse_lstm = np.sqrt(mse)
        return self.rmse_lstm,

    def eval_model_RNN(self, params):
        '''
        Gera uma avaliação do individuo - utilizando uma RNN LSTM Pytorch
        :param params: Recebe os parametros do individuo no DEAP
        :return: RMSE do Individuo
        '''

        input_dim = len(self.X_train.columns)
        output_dim = 1
        self.params = params
        #hidden_dim = 64
        hidden_dim = self.hidden_layer[self.params[0]]
        #layer_dim = 3
        layer_dim = 3
        batch_size = 64
        #batch_size = self.hidden_layer[self.params[0]]
        #dropout = 0.2
        dropout = self.batch_size[self.params[3]]
        n_epochs = 20
        #learning_rate = 1e-3
        learning_rate = self.learning_rate[self.params[4]]
        #weight_decay = 1e-6
        weight_decay = self.alpha[self.params[2]]
        model_params = {'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'layer_dim': layer_dim,
                        'output_dim': output_dim,
                        'dropout_prob': dropout}
        model = self.get_model('rnn', model_params)

        loss_fn = nn.MSELoss(reduction="mean")
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 0.548

        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(self.train_loader, self.val_loader, batch_size=batch_size,
                  n_epochs=n_epochs, n_features=input_dim)
        #opt.plot_losses()

        predictions, values = opt.evaluate(
            self.test_loader_one,
            batch_size=1,
            n_features=input_dim
        )
        scaler = self.get_scaler('minmax')
        scaler.fit(self.y_train.values.reshape(-1,1))
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds},
                                 index=self.X_test.head(len(vals)).index)
        df_result = df_result.sort_index()
        df_result = self.inverse_transform(scaler, df_result, [["value", "prediction"]])
        #print(df_result)
        mse = (np.square(df_result.value.values - df_result.prediction.values)).mean(axis=None)
        self.rmse_rnn = np.sqrt(mse)
        return self.rmse_rnn,

    def eval_model_MLP(self, params, full=False):
        '''
        Gera uma avaliação do individuo - utilizando uma RNN LSTM Pytorch
        :param params: Recebe os parametros do individuo no DEAP
        :return: RMSE do Individuo
        '''

        input_dim = len(self.X_train.columns)
        output_dim = 1
        self.params = params
        #hidden_dim = 64
        hidden_dim = self.hidden_layer[self.params[0]]
        #layer_dim = 3
        layer_dim = 3
        batch_size = 64
        #batch_size = self.hidden_layer[self.params[0]]
        #dropout = 0.2
        dropout = self.batch_size[self.params[3]]
        n_epochs = 20
        #learning_rate = 1e-3
        learning_rate = self.learning_rate[self.params[4]]
        #weight_decay = 1e-6
        weight_decay = self.alpha[self.params[2]]
        func_act = self.activation[self.params[1]]
        op = self.max_iter[self.params[5]]
        seed = self.fator_aleatorio[self.params[6]]
        model_params = {'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'layer_dim': layer_dim,
                        'output_dim': output_dim,
                        'dropout_prob': dropout,
                        'func_act': func_act,
                        'op': op,
                        'seed': seed}
        model = self.get_model('mlp', model_params)

        loss_fn = nn.MSELoss(reduction="mean")
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = op(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 0.548
        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(self.train_loader, self.val_loader, batch_size=batch_size,
                  n_epochs=n_epochs, n_features=input_dim)
        #opt.plot_losses()

        predictions, values = opt.evaluate(
            self.test_loader_one,
            batch_size=1,
            n_features=input_dim
        )
        scaler = self.get_scaler('minmax')
        scaler.fit(self.y_train.values.reshape(-1,1))
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds},
                                 index=self.X_test.head(len(vals)).index)
        df_result = df_result.sort_index()
        df_result = self.inverse_transform(scaler, df_result, [["value", "prediction"]])
        if full:
            inicio_predicao = datetime.datetime.now()
            predictions, values = opt.evaluate(
                self.full_loader_one,
                batch_size=1,
                n_features=input_dim
            )
            fim_predicao = datetime.datetime.now()
            print(f"Duracao da Previsao DEAP/MLP: {fim_predicao - inicio_predicao}")
            scaler = self.get_scaler('minmax')
            scaler.fit(self.y.values.reshape(-1,1))
            vals = np.concatenate(values, axis=0).ravel()
            preds = np.concatenate(predictions, axis=0).ravel()
            self.df_result_full = pd.DataFrame(data={"value": vals, "prediction": preds},
                                               index=self.X.index)
            self.df_result_full = self.df_result_full.sort_index()
            self.df_result_full = self.inverse_transform(scaler, self.df_result_full, [["value", "prediction"]])
            return self
        #print(df_result)
        mse = (np.square(df_result.value.values - df_result.prediction.values)).mean(axis=None)
        self.rmse_mlp = np.sqrt(mse)
        return self.rmse_mlp,

    def eval_model_CONV1D(self, params, full=False):
        '''
        Gera uma avaliação do individuo - utilizando uma RNN LSTM Pytorch
        :param params: Recebe os parametros do individuo no DEAP
        :return: RMSE do Individuo
        '''

        input_dim = len(self.X_train.columns)
        output_dim = 1
        self.params = params
        #hidden_dim = 64
        hidden_dim = self.hidden_layer[self.params[0]]
        #layer_dim = 3
        layer_dim = 3
        batch_size = 64
        #batch_size = self.hidden_layer[self.params[0]]
        #dropout = 0.2
        dropout = self.batch_size[self.params[3]]
        n_epochs = 20
        #learning_rate = 1e-3
        learning_rate = self.learning_rate[self.params[4]]
        #weight_decay = 1e-6
        weight_decay = self.alpha[self.params[2]]
        func_act = self.activation[self.params[1]]
        op = self.max_iter[self.params[5]]
        seed = self.fator_aleatorio[self.params[6]]
        model_params = {'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'layer_dim': layer_dim,
                        'output_dim': output_dim,
                        'dropout_prob': dropout,
                        'func_act': func_act,
                        'op': op,
                        'seed': seed}
        model = self.get_model('conv1d', model_params)

        loss_fn = nn.MSELoss(reduction="mean")
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer = op(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # 0.548
        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
        opt.train(self.train_loader, self.val_loader, batch_size=batch_size,
                  n_epochs=n_epochs, n_features=input_dim)
        #opt.plot_losses()

        predictions, values = opt.evaluate(
            self.test_loader_one,
            batch_size=1,
            n_features=input_dim
        )
        scaler = self.get_scaler('minmax')
        scaler.fit(self.y_train.values.reshape(-1,1))
        vals = np.concatenate(values, axis=0).ravel()
        preds = np.concatenate(predictions, axis=0).ravel()
        df_result = pd.DataFrame(data={"value": vals, "prediction": preds},
                                 index=self.X_test.head(len(vals)).index)
        df_result = df_result.sort_index()
        df_result = self.inverse_transform(scaler, df_result, [["value", "prediction"]])
        if full:
            inicio_predicao = datetime.datetime.now()
            predictions, values = opt.evaluate(
                self.full_loader_one,
                batch_size=1,
                n_features=input_dim
            )
            fim_predicao = datetime.datetime.now()
            print(f"Duracao da Previsao DEAP/Conv: {fim_predicao - inicio_predicao}")
            scaler = self.get_scaler('minmax')
            scaler.fit(self.y.values.reshape(-1,1))
            vals = np.concatenate(values, axis=0).ravel()
            preds = np.concatenate(predictions, axis=0).ravel()
            self.df_result_full = pd.DataFrame(data={"value": vals, "prediction": preds},
                                               index=self.X.index)
            self.df_result_full = self.df_result_full.sort_index()
            self.df_result_full = self.inverse_transform(scaler, self.df_result_full, [["value", "prediction"]])
            return self
        #print(df_result)
        mse = (np.square(df_result.value.values - df_result.prediction.values)).mean(axis=None)
        self.rmse_rnn = np.sqrt(mse)
        return self.rmse_rnn,

    def deap_creator_toolbox(self, eval_clf, mlp=False, n_pop = 100, ger=10):

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        if mlp:
            toolbox.register("camada_escondida", random.sample,
                             range(len(self.hidden_layer)), len(self.hidden_layer))
            toolbox.register("alpha", random.sample,
                             range(len(self.alpha)), len(self.alpha))
            toolbox.register("funcao_ativacao", random.sample,
                             range(len(self.activation)), len(self.activation))
            toolbox.register("tamanho_lote", random.sample,
                             range(len(self.batch_size)), len(self.batch_size))
            toolbox.register("taxa_aprendizado", random.sample,
                             range(len(self.learning_rate)), len(self.learning_rate))
            toolbox.register("numero_max_it", random.sample,
                             range(len(self.max_iter)), len(self.max_iter))
            toolbox.register("fator_aleatorio", random.sample,
                             range(len(self.fator_aleatorio)), len(self.fator_aleatorio))
            # Structure initializers
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.camada_escondida)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.alpha)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.funcao_ativacao)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.tamanho_lote)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.taxa_aprendizado)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.numero_max_it)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.fator_aleatorio)
            toolbox.register("population", tools.initRepeat, list,
                             toolbox.individual)
        else:
            toolbox.register("camada_0", random.sample,
                             range(len(self.tamanho_camada)), len(self.tamanho_camada))
            toolbox.register("activation_function", random.sample,
                             range(len(self.activation_keras)), len(self.activation_keras))
            toolbox.register("optimizers", random.sample,
                             range(len(self.optimizers)), len(self.optimizers))
            toolbox.register("param_vazio_1", random.sample,
                             range(len(self.param_vazio_1)), len(self.param_vazio_1))
            toolbox.register("param_vazio_2", random.sample,
                             range(len(self.param_vazio_2)), len(self.param_vazio_2))
            toolbox.register("param_vazio_3", random.sample,
                             range(len(self.param_vazio_3)), len(self.param_vazio_3))
            toolbox.register("param_vazio_4", random.sample,
                             range(len(self.param_vazio_4)), len(self.param_vazio_4))
            toolbox.register("param_vazio_5", random.sample,
                             range(len(self.param_vazio_5)), len(self.param_vazio_5))
            toolbox.register("param_vazio_6", random.sample,
                             range(len(self.param_vazio_6)), len(self.param_vazio_6))
            # Structure initializers
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.camada_0)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.activation_function)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.optimizers)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.param_vazio_1)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.param_vazio_2)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.param_vazio_3)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.param_vazio_4)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.param_vazio_5)
            toolbox.register("individual", tools.initIterate,
                             creator.Individual,
                             toolbox.param_vazio_6)
            toolbox.register("population", tools.initRepeat, list,
                             toolbox.individual)
        #print('Aqui ainda tá indo, Passou o Toolbox')
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        #toolbox.register("select", tools.selNSGA2)
        toolbox.register("select", tools.selBest)
        toolbox.register("evaluate", eval_clf)
        #print('Aqui ainda tá indo, Passou todo o Toolbox')
        pop = toolbox.population(n=n_pop)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        #print('Passou os pop hof e stats!!!')
        #scores.append(ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        #print('registrou o stats!!!')
        #print(stats)
        pop, logbook = algorithms.eaMuPlusLambda(pop,
                                                 toolbox,
                                                 mu=n_pop,
                                                 lambda_=n_pop,
                                                 cxpb=0.4,
                                                 mutpb=0.5, ngen=ger,
                                                 stats=stats,
                                                 halloffame=hof,
                                                 verbose=True)
        #print('Terminou a function!!!')
        return pop, stats, hof, logbook

    def hof_params_mlp(self, hof):
        '''
        Parametros que obtiveram melhores resultados = HOF
        '''
        for h in hof:
            print(f'Quantidade camadas escondidas: {self.hidden_layer[h[0]]}')
            print(f'Função de Ativação: {str(self.activation[h[1]])}')
            print(f'Taxa de aprendizado: {self.learning_rate[h[4]]}')
            print(f'Otimizador: {str(self.max_iter[h[5]])}')
            print(f'Peso: {self.alpha[h[2]]}')
            print(f'Taxa de dropout: {self.batch_size[h[3]]}')
            print(f'Fator aleatório: {self.fator_aleatorio[h[6]]}')

    def grafico_evolucionario(self, logbook, benchmarking):
        '''
        Imprime um gráfico com a evolução das avaliaçoes pelas geraçoes determinadas
        :param logbook: Dados do processo evolucionário
        :param benchmarking: Valor de referência a ser batido
        :return: Gráfico com Métrica de avaliaçao dos melhores indíviduos pelas geracoes e a média da
        populaçao
        '''
        self.logbook = logbook
        gen = self.logbook.select("gen")
        #fit_max = self.logbook.select("max")
        fit_mins = self.logbook.select("min")
        size_avgs = self.logbook.select("avg")
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, fit_mins, "b-", label="Melhor Score da Pop")
        ax1.set_xlabel("# Gerações")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")
        ax2 = ax1.twinx()
        line2 = ax2.plot(gen, size_avgs, "r--", label="Score Médio")
        ax2.set_ylabel("Média da População", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")
        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.set_ylim(min(fit_mins)*0.995,
                     max(fit_mins)*1.001)
        ax2.set_ylim(min(fit_mins)*0.995,
                     max(fit_mins)*1.001)
        ax1.legend(lns, labs, loc="best")
        ax1.axhline(benchmarking, color='g', linestyle=':',
                    label='Benchmark')
        return plt.show()

def series_to_supervised(df, n_out=1, dropnan=True):
    '''
    Do algoritmo de experimento de seleção de caracterisitcas.py
    :param n_in: Número de observaçoes defasadas(lag) do input
    :param n_out: Número de observaçoes defasadas(lag) do output
    :param dropnan:
    :return: Dataframe adequado para supervised learning
    '''
    df_sup = df
    for col in df_sup.columns:
        for j in range(0, n_out):
            col_name_2 = str(col+f'_(t-{j})')
            df_sup[col_name_2] = df_sup[col].shift(-j)
        df_sup = df_sup.drop(col, axis=1)
    if dropnan:
        df_sup = df_sup.dropna() # data cleaning
    return df_sup

def grafico_dispersao_histograma(df, nome_modelo:str):
    erro = df.value.values - df.prediction.values
    fig, axs = plt.subplots(1, 2, sharey=False, tight_layout=True)
    axs[1].hist(erro, bins=50)
    axs[1].set_title(f'Histograma de Residuos - {nome_modelo}')
    axs[0].scatter(df.value.values, df.prediction.values)
    axs[0].set_ylim([5, 14])
    axs[0].set_title(f'Real x Predito - {nome_modelo}')
    return plt.show()

def plot_graf(df):
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.plot(df.value.values, label='Real')
    plt.plot(df.prediction.values, label='Modelo DEAP/LSTM')
    plt.title('Real x Modelo DEAP/LSTM')
    plt.legend(loc='best')
    return plt.show()

if __name__ == "__main__":
    # Base de Dados Crua
    df = pd.read_excel('dados_artigo_2 - cópia.xlsx', index_col=0).dropna()
    #df = pd.read_excel('dados_artigo_2019.xlsx', index_col=0)
    df = df.fillna(df.median())
    #print(df.head())

    # Base de Dados adaptada para um problema de aprendizado supervisionado
    df_sup = series_to_supervised(df, 6) # 6 Periodos de 4h = 24h
    #print(df_sup)

    # Características selecionadas no artigo para dados de entrada
    feat_selected = ['CEG_(t-1)', 'Prod_PQ_(t-0)', 'Cfix_(t-0)',
                     'pos_dp_03_(t-0)', 'Temp_02_(t-0)', 'Temp_05_(t-0)',
                     'Pres_01_(t-0)', 'Pres_04_(t-0)', 'rpm_03_(t-0)',
                     'Alt_Cam_(t-5)', 'Pres_01_(t-5)','rpm_06_(t-5)']
    #X = df_sup[feat_selected]
    X = df_sup.drop('CEG_(t-0)', axis=1)
    y = df_sup['CEG_(t-0)']
    deap = DEAP(X, y)
    # deap.train_teste_base_torch()
    # start = datetime.datetime.now()
    # for _ in range(5):
    print('-'*40+' LSTM '+'-'*40)
    # print(start)
    # eval_clf = data_analysis.eval_model_voting_clf
    # eval_clf = deap.eval_model_LSTM
    # pop, stats, hof, logbook = deap.deap_creator_toolbox(eval_clf, mlp=True, n_pop=250, ger=10)
    # print('-'*30+' Melhores Parametros em LSTM '+'-'*30)
    # deap.hof_params_mlp(hof)
    # # print('-'*80)
    # # # deap.grafico_evolucionario(logbook, 0.54)
    # for h in hof:
    #     df_result_lstm = deap.eval_model_LSTM(h, full=True).df_result_full
    # # print(df)
    # #grafico_dispersao_histograma(df_result_lstm, 'LSTM')
    # plot_graf(df_result_lstm)
    # end = datetime.datetime.now()
    # print(end)
    # print(f'Duração LSTM: {end - start}')
        # print('\n')
        # print('-'*40+' MLP '+'-'*40)
        # start = datetime.datetime.now()
        # print(start)
    # eval_clf = data_analysis.eval_model_voting_clf
    # eval_clf = deap.eval_model_RNN
    # pop, stats, hof, logbook = deap.deap_creator_toolbox(eval_clf, mlp=True, n_pop=10, ger=10)
    # end = datetime.datetime.now()
    # print(end)
    # print(f'Duração RNN: {end - start}')
    # print('\n')
    # start = datetime.datetime.now()
    # print(start)
    # eval_clf = data_analysis.eval_model_voting_clf
    # eval_clf = deap.eval_model_MLP
    # pop, stats, hof, logbook = deap.deap_creator_toolbox(eval_clf, mlp=True, n_pop=300, ger=10)
    # print('-'*30+' Melhores Parametros em MLP '+'-'*30)
    # deap.hof_params_mlp(hof)
    # print('-'*80)
    # # deap.grafico_evolucionario(logbook, 0.54)
    # for h in hof:
    #     df_result_mlp = deap.eval_model_MLP(h, full=True).df_result_full
    # print(df)
    # grafico_dispersao_histograma(df, 'MLP')
    # end = datetime.datetime.now()
    # print(end)
    # print(f'Duração MLP: {end - start}')
    # print('\n')
    print('-'*40+' Conv1d '+'-'*40)
    deap.train_teste_base_torch(cv1d=True) # A camada convulotional pede uma mudança de formato dos dados de entrada
    start = datetime.datetime.now()
    print(start)
    # eval_clf = data_analysis.eval_model_voting_clf
    eval_clf = deap.eval_model_CONV1D
    pop, stats, hof, logbook = deap.deap_creator_toolbox(eval_clf, mlp=True, n_pop=10, ger=10)
    print('-'*40+' Melhores Parametros em Conv1d '+'-'*40)
    deap.hof_params_mlp(hof)
    print('-'*80)
    # deap.grafico_evolucionario(logbook, 0.54)
    for h in hof:
        df_result_conv1d = deap.eval_model_CONV1D(h, full=True).df_result_full
    # print(df)
    # grafico_dispersao_histograma(df, 'Conv 1d')
    end = datetime.datetime.now()
    print(end)
    print(f'Duração Conv1d: {end - start}')
    # grafico_dispersao_histograma(df_result_lstm, 'LSTM')
    # grafico_dispersao_histograma(df_result_mlp, 'MLP')
    # grafico_dispersao_histograma(df_result_conv1d, 'Conv 1d')


