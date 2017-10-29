from mnist import MNIST
from random import *

mndata = MNIST('data')

images, labels = mndata.load_training()

index = 2  # choose an index

# print(mndata.display(images[index]))
#print(mndata.train_images[index])
# print(mndata.train_labels[index])
# print(len(mndata.train_images))
# print(dir(mndata))



class Brain:
    def __init__(self):
        self.input_layer = Layer(256)
        self.hidden_layer = Layer(16)
        self.output_layer = Layer(10)
        self.input_layer.attach(self.hidden_layer)
        self.hidden_layer.attach(self.output_layer)
    def process(self, image):
        self.input_layer.process(image)
        self.hidden_layer.process("none")
        self.output_layer.process("none")
        self.output_layer.render()

class Layer:
    def __init__(self, count):
        self.neurons = []
        i = 0
        while i < count:
            self.neurons.append(Neuron())
            i += 1
    def attach(self, target_layer):
        for idx, self_neuron in enumerate(self.neurons):
            for idx, target_neuron in enumerate(target_layer.neurons):
                self_neuron.synapse_onto(target_neuron)
    def process(self, image):
        for idx, neuron in enumerate(self.neurons):
            if image == "none":
                neuron.receive_input("none")
            else:
                neuron.receive_input(image[idx])
    def render(self):
        for idx, neuron in enumerate(self.neurons):
            print(neuron.activity)

class Neuron:
    def __init__(self):
        self.outputs = []
        self.inputs = []
        self.activity = 0.0
        self.target = 0.0
    def receive_input(self, brightness):
        if brightness == "none": ##Sum activity
            constant_factor = 16.0 / len(self.inputs)
            add_up = 0.0
            for idx, input_synapse in self.inputs:
                add_up += input_synapse.get_act()*constant_factor
            if add_up > 1.0:
                add_up = 1.0
            if add_up < 0: ##Bias toward sparse coding format
                add_up = 0
            self.activity = add_up
        else:
            self.activity = (brightness/128.0) - 1.0
    def synapse_onto(self, neuron):
        syn = Synapse(self, neuron)
        self.outputs.append(syn)
        neuron.inputs.append(syn)
    def calc_error(self, target):
        err = self.activity - self.target
        for idx, input_synapse in self.inputs:
            presynaptic = input_synapse.presynaptic
            effect = (presynaptic.activity - self.activity) * (input_synapse.weight)
            input_synapse.change_weight(err * effect * -1)

class Synapse:
    def __init__(self, neuron1, neuron2):
        self.weight = random()*2.0 - 1.0
        self.presynaptic = neuron1
        self.postsynaptic = neuron2
        neuron1.outputs.append(self)
        neuron2.inputs.append(self)
    def get_act(self):
        return self.weight * self.presynaptic.activity
    def change_weight(self, amount):
        delta = 100.0
        self.weight -= delta * amount
        if self.weight > 1.0:
            self.weight = 1.0
        if self.weight < -1.0:
            self.weight = -1.0


x = Brain()
x.process(mndata.train_images[index])



#
