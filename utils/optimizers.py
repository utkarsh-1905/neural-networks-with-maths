import numpy as np


class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            # with momentum
            if not hasattr(layer, 'weight_momentum'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)

            weight_update = self.momentum*layer.weight_momentum-self.current_lr*layer.dweights
            layer.weight_momentum = weight_update

            bias_update = self.momentum*layer.bias_momentum-self.current_lr*layer.dbiases
            layer.bias_momentum = bias_update
        else:
            # vanilla sgd
            weight_update = -self.current_lr * layer.dweights
            bias_update = -self.current_lr * layer.dbiases

        # update weights and biases with/without momentum
        layer.weights += weight_update
        layer.biases += bias_update

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.iterations = 0
        self.epsilon = epsilon
        self.decay = decay

    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        layer.weights += -self.current_lr*layer.dweights / \
            (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_lr*layer.dbiases / \
            (np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_RMSProp:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.rho = rho
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.iterations = 0
        self.decay = decay

    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + \
            (1-self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + \
            (1-self.rho) * layer.dbiases**2

        layer.weights += -self.current_lr * layer.dweights / \
            (np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / \
            (np.sqrt(layer.bias_cache)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class Optimizer_Adam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.iterations = 0
        self.decay = decay
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * \
                (1./(1+self.decay*self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * \
            layer.weight_momentums+(1-self.beta_1)*layer.dweights
        layer.bias_momentums = self.beta_1 * \
            layer.bias_momentums + (1-self.beta_1)*layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1-self.beta_1**(self.iterations+1))
        bias_momentum_corrected = layer.bias_momentums / \
            (1-self.beta_1 ** (self.iterations+1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1-self.beta_2)*layer.dweights**2
        layer.bias_cache = self.beta_2*layer.bias_cache + \
            (1-self.beta_2)*layer.dbiases**2

        weight_cache_corrected = layer.weight_cache / \
            (1-self.beta_2**(self.iterations+1))
        bias_cache_corrected = layer.bias_cache / \
            (1-self.beta_2**(self.iterations+1))

        layer.weights += -self.current_lr*weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected)+self.epsilon)
        layer.biases += -self.current_lr*bias_momentum_corrected / \
            (np.sqrt(bias_cache_corrected)+self.epsilon)

    def post_update_params(self):
        self.iterations += 1
