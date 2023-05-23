import numpy as np
import time
import random
import logging
from pathlib import Path

random.seed(int(time.time()))
np.random.seed(int(time.time()))

import numpy as np
import matplotlib.pyplot as plt
import time
import random

import torch.nn.functional as F

random.seed(int(time.time()))
np.random.seed(int(time.time()))


class MLP():
    def __init__(self, train_dl, test_dl, epoch, learning_rate, gamma=1, initialization="Xavier", hidden_nodes=20, log_name=None, gradient_descent_strategy="SGD", data_dim=784, label_dim=10):
        # Gradient Descent strategy
        self.gradient_descent_strategy = gradient_descent_strategy
        self.log_name=log_name

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma  # learning_rate decay hyperparameter gamma
        self.epoch = epoch
        self.data_dim = data_dim
        self.label_dim = label_dim
        self.hidden_nodes = hidden_nodes
        self.initialization = initialization

        # Metrics
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []

        # Dataloader
        self.train_dl = train_dl
        self.test_dl = test_dl
        # Inter Variable like z1, a1, z2
        self.inter_variable = {}

        # Gradient Descent Parameter
        self.momentum_v_layer1 = 0
        self.momentum_v_layer2 = 0
        self.momentum_beta = 0.9

        # RMSprop hyperparameters can use larger learning rate
        self.RMS_s_layer1 = 0
        self.RMS_s_layer2 = 0
        self.RMS_beta = 0.999
        self.RMS_epsilon = 1e-8

        # Adam hyperparameters
        self.Adam_v_layer1 = 0
        self.Adam_v_layer2 = 0
        self.Adam_s_layer1 = 0
        self.Adam_s_layer2 = 0
        self.Adam_beta1 = 0.9
        self.Adam_beta2 = 0.999
        self.Adam_epsilon = 1e-8

        if self.log_name != None:
            self.create_log()

    def initialize_weights(self):
        if self.initialization == "Xavier":
            w1 = np.random.randn(self.data_dim, self.hidden_nodes) * np.sqrt(6/(1+self.data_dim+self.hidden_nodes))
            w2 = np.random.randn(self.hidden_nodes, self.label_dim) * np.sqrt(6/(1+self.hidden_nodes+self.label_dim))
        elif self.initialization == "He":
            w1 = np.random.randn(self.data_dim, self.hidden_nodes) * np.sqrt(2/self.data_dim)
            w2 = np.random.randn(self.hidden_nodes, self.label_dim) * np.sqrt(2/self.hidden_nodes)
        elif self.initialization == "Gaussian":
            w1 = np.random.randn(self.data_dim, self.hidden_nodes)
            w2 = np.random.randn(self.hidden_nodes, self.label_dim)
        elif self.initialization == "Random":
            w1 = np.random.uniform(-1, 1, (self.data_dim, self.hidden_nodes))
            w2 = np.random.uniform(-1, 1, size=(self.hidden_nodes, self.label_dim))
        elif self.initialization == "Constant0":
            w1 = np.zeros((self.data_dim, self.hidden_nodes))
            w2 = np.zeros((self.hidden_nodes, self.label_dim))
        else:
            raise NotImplemented
        return w1, w2

    def train(self, optimizer, activation, gradient_check=False):
        start = time.time()
        w1, w2 = self.initialize_weights()

        for j in range(self.epoch):
            ema_train_accuracy = None
            ema_train_loss = None

            for step, data in enumerate(self.train_dl):
                learning_rate = self.learning_rate
                train_data, train_labels = data
                train_data = train_data.view(train_data.shape[0], -1).numpy().T
                train_labels = F.one_hot(train_labels).numpy()
                if self.gradient_descent_strategy == "SGD":
                        # forward feed
                        self.forward(x=train_data, w1=w1, w2=w2, no_gradient=False, activation=activation)
                        # Calculate gradient
                        gradient1, gradient2 = self.back_prop(x=train_data, y=train_labels, w1=w1, w2=w2, activation=activation)
                        w1, w2, learning_rate = self.update_weight(w1, w2, gradient1, gradient2, optimizer=optimizer, epoch=j + 1, learning_rate=learning_rate)
                        train_accuracy = self.accuracy(train_labels, self.inter_variable["z2"])
                        train_loss = self.loss(self.inter_variable["z2"], train_labels)

                        # Gradient check if required
                        if gradient_check:
                            self.gradient_check(train_data, train_labels, w1, w2, gradient1, gradient2, activation=activation)

                        if ema_train_accuracy is not None:
                            ema_train_accuracy = ema_train_accuracy * 0.98 + train_accuracy * 0.02
                            ema_train_loss = ema_train_loss * 0.98 + train_loss * 0.02

                        else:
                            ema_train_accuracy = train_accuracy
                            ema_train_loss = train_loss
                        if step % 50 == 0:
                            self.log_string(f'Train:Step/Epoch:{step}/{j}, Accuracy:{train_accuracy*100:.2f}, Loss:{train_loss:.4f}')
                else:
                    raise NotImplemented

            # Evaluate
            temp_test_accuracy = []
            temp_test_loss = []
            for step, data in enumerate(self.test_dl):
                test_data, test_labels = data
                test_data = test_data.view(test_data.shape[0], -1).numpy().T
                test_labels = F.one_hot(test_labels).numpy()

                test_forward = self.forward(test_data, w1, w2, no_gradient=True, activation=activation)
                test_accuracy = self.accuracy(test_labels, test_forward)
                test_loss = self.loss(test_forward, test_labels)
                temp_test_accuracy.append(test_accuracy)
                temp_test_loss.append(test_loss)

            current_test_accuracy = np.mean(temp_test_accuracy)
            current_test_loss = np.mean(temp_test_loss)
            self.log_string(f"Epoch:{j + 1}")
            self.log_string(f"Test: Accuracy: {(100 * current_test_accuracy):.2f}%, Loss: {current_test_loss:.4f}")
            # for plot
            self.train_accuracy.append(ema_train_accuracy)
            self.train_loss.append(ema_train_loss)
            self.test_accuracy.append(current_test_accuracy)
            self.test_loss.append(current_test_loss)

        end = time.time()
        self.log_string(f"Trained time: {(end - start)} s")
        return np.asarray(self.test_accuracy), np.asarray(self.train_loss)

    def forward(self, x, w1, w2, no_gradient: bool, activation):
        """
        :param x: Input Data
        :param no_gradient: distinguish it's train or evaluate
        :return: if no_gradient = False, return output
        """

        if activation == "Tanh":
            z1 = w1.T.dot(x)
            a1 = np.tanh(z1)
            z2 = w2.T.dot(a1)
        elif activation == "ReLU":
            z1 = w1.T.dot(x)
            a1 = np.maximum(0, z1)
            z2 = w2.T.dot(a1)
        elif activation == "Sigmoid":
            z1 = w1.T.dot(x)
            a1 = 1 / (1 + np.exp(-z1))
            z2 = w2.T.dot(a1)

        if no_gradient:
            # for predict
            return z2
        else:
            # For back propagation
            self.inter_variable = {"z1": z1, "a1": a1, "z2": z2}

    def back_prop(self, x, y, w1, w2, activation):
        """
        :param i: for Adam bias correction
        """
        m = x.shape[1]

        #########################################################################################
        #                           code you need to fill
        #  Pay attention to matrix shape in all codes
        if activation == "Tanh":
            delta_k = self.inter_variable["z2"] - y.T
            delta_j = (1 - self.inter_variable["a1"] ** 2) * (w2.dot(delta_k))
            gradient1 = 1. / m * (x.dot(delta_j.T))
            gradient2 = 1. / m * (self.inter_variable["a1"].dot(delta_k.T))
            return gradient1, gradient2
        elif activation == "Sigmoid":
            delta_k = self.inter_variable["z2"] - y.T
            delta_j = self.inter_variable["a1"] * (1 - self.inter_variable["a1"]) * w2.dot(delta_k)
            gradient1 = 1. / m * (x.dot(delta_j.T))
            gradient2 = 1. / m * (self.inter_variable["a1"].dot(delta_k.T))
            return gradient1, gradient2
        elif activation == "ReLU":
            delta_k = (self.inter_variable["z2"] - y.T)
            delta_relu = self.inter_variable["a1"]
            delta_relu[delta_relu <= 0] = 0
            delta_relu[delta_relu > 0] = 1
            delta_j = delta_relu * (w2.dot(delta_k))
            gradient1 = 1. / m * (x.dot(delta_j.T))
            gradient2 = 1. / m * (self.inter_variable["a1"].dot(delta_k.T))
            return gradient1, gradient2
        #########################################################################################

    def update_weight(self, w1, w2, gradient1, gradient2, optimizer, epoch, learning_rate):
        if optimizer == "SGD":
            return self.SGD(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "Momentum":
            return self.Momentum(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "RMSprop":
            return self.RMSprop(w1, w2, gradient1, gradient2, learning_rate)
        elif optimizer == "Adam":
            return self.Adam(epoch, w1, w2, gradient1, gradient2, learning_rate)

    def SGD(self, w1, w2, gradient1, gradient2, learning_rate):
        w1 -= learning_rate * gradient1
        w2 -= learning_rate * gradient2
        # Learning rate decay
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    def Momentum(self, w1, w2, gradient1, gradient2, learning_rate):
        self.momentum_v_layer1 = self.momentum_beta * self.momentum_v_layer1 + (1 - self.momentum_beta) * gradient1
        self.momentum_v_layer2 = self.momentum_beta * self.momentum_v_layer2 + (1 - self.momentum_beta) * gradient2
        w1 -= learning_rate * self.momentum_v_layer1
        w2 -= learning_rate * self.momentum_v_layer2
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    def RMSprop(self, w1, w2, gradient1, gradient2, learning_rate):
        self.RMS_s_layer1 = self.RMS_beta * self.RMS_s_layer1 + (1 - self.RMS_beta) * gradient1 ** 2
        self.RMS_s_layer2 = self.RMS_beta * self.RMS_s_layer2 + (1 - self.RMS_beta) * gradient2 ** 2
        w1 -= self.learning_rate * gradient1 / (np.sqrt(self.RMS_s_layer1) + self.RMS_epsilon)
        w2 -= self.learning_rate * gradient2 / (np.sqrt(self.RMS_s_layer2) + self.RMS_epsilon)
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    def Adam(self, t, w1, w2, gradient1, gradient2, learning_rate):
        # Momentum part
        self.Adam_v_layer1 = self.Adam_beta1 * self.Adam_v_layer1 + (1 - self.Adam_beta1) * gradient1
        self.Adam_v_layer2 = self.Adam_beta1 * self.Adam_v_layer2 + (1 - self.Adam_beta1) * gradient2
        # RMS part
        self.Adam_s_layer1 = self.Adam_beta2 * self.Adam_s_layer1 + (1 - self.Adam_beta2) * gradient1 ** 2
        self.Adam_s_layer2 = self.Adam_beta2 * self.Adam_s_layer2 + (1 - self.Adam_beta2) * gradient2 ** 2
        # Bias correction
        Adam_v_layer1_corrected = self.Adam_v_layer1 / (1 - self.Adam_beta1 ** t)
        Adam_v_layer2_corrected = self.Adam_v_layer2 / (1 - self.Adam_beta1 ** t)
        Adam_s_layer1_corrected = self.Adam_s_layer1 / (1 - self.Adam_beta2 ** t)
        Adam_s_layer2_corrected = self.Adam_s_layer2 / (1 - self.Adam_beta2 ** t)
        # Update weights
        w1 -= self.learning_rate * Adam_v_layer1_corrected / (
                np.sqrt(Adam_s_layer1_corrected) + self.Adam_epsilon)
        w2 -= self.learning_rate * Adam_v_layer2_corrected / (
                np.sqrt(Adam_s_layer2_corrected) + self.Adam_epsilon)
        learning_rate *= self.gamma
        return w1, w2, learning_rate

    @staticmethod
    def accuracy(label, y_hat: np.ndarray):
        y_hat = y_hat.T
        acc = y_hat.argmax(axis=1) == label.argmax(axis=1)
        b = acc + 0
        return b.mean()

    def save(self, filename):
        np.savez(filename, self.weights1_list, self.weights2_list)

    @staticmethod
    def loss(output, label):
        # Loss = 1/n * 1/2 * ∑(yk - tk)^2
        a = label.shape[0]
        return np.sum(((output.T - label) ** 2)) / (2 * label.shape[0])

    def gradient_check(self, x, y, w1, w2, gradient1, gradient2, activation, epsilon=1e-7):
        parameters = np.vstack((w1.reshape((100, 1)), w2.reshape((60, 1))))
        grad = np.vstack((gradient1.reshape((100, 1)), gradient2.reshape(60, 1)))
        num_parameters = parameters.shape[0]
        gradapprox = np.zeros((num_parameters, 1))
        J_plus = np.zeros((num_parameters, 1))
        J_minus = np.zeros((num_parameters, 1))
        for i in range(num_parameters):
            thetaplus = np.copy(parameters)
            thetaplus[i][0] = thetaplus[i][0] + epsilon
            w_plus_layer1 = thetaplus[0:100].reshape(5, 20)
            w_plus_layer2 = thetaplus[100:160].reshape(20, 3)
            J_plus[i] = self.evaluate(x, y, w_plus_layer1, w_plus_layer2, activation)

            thetaminus = np.copy(parameters)
            thetaminus[i][0] = thetaminus[i][0] - epsilon
            w_minus_layer1 = thetaminus[0:100].reshape(5, 20)
            w_minus_layer2 = thetaminus[100:160].reshape(20, 3)
            J_minus[i] = self.evaluate(x, y, w_minus_layer1, w_minus_layer2, activation)
            gradapprox[i] = (J_plus[i] - J_minus[i]) / (2. * epsilon)
        numerator = np.linalg.norm(grad - gradapprox)
        denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
        difference = numerator / denominator
        print(f"L2 distance of Gradient check:{difference}")

    def evaluate(self, x, y, w1, w2, activation):
        z1 = w1.T.dot(x)
        if activation == "Tanh":
            a1 = np.tanh(z1)
        elif activation == "ReLU":
            a1 = np.maximum(0, z1)
        elif activation == "Sigmoid":
            a1 = 1 / (1 + np.exp(-z1))
        z2 = w2.T.dot(a1)
        return np.sum(((z2.T - y) ** 2) / (2 * y.shape[0]))

    def plot_test(self):
        plt.figure(facecolor='w',edgecolor='w')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(self.test_accuracy, label="Test Accuracy", alpha=0.5)
        plt.xticks(np.arange(0, len(self.test_accuracy)) )
        plt.legend()
        if self.log_name != None:
            plt.savefig('./'+self.log_name+'/accuracy.png', dpi=600, format='png')

    def plot_loss(self):
        plt.figure(facecolor='w',edgecolor='w')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(np.array(self.train_loss), label="Train Loss", alpha=0.5)
        plt.xticks(np.arange(0, len(self.train_loss)))
        plt.legend()
        if self.log_name != None:
            plt.savefig('./'+self.log_name+'/loss.png', dpi=600, format='png')

    def create_log(self):
        log_dir = Path('./'+self.log_name+'/')
        log_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger("MLP")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler('%s/%s_log.txt' % (log_dir, "mlp")) 
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_string(self, str):
        if self.log_name != None:
            self.logger.info(str)
        print(str)