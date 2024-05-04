import mnist_loader
import mnist_NN
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)

net = mnist_NN.Network([784, 30, 10])
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)