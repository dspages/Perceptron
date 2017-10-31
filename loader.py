from mnist import MNIST
from random import *
import pdb;

mndata = MNIST('data')

images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# print(mndata.display(images[index]))
#print(mndata.train_images[index])
# print(mndata.train_labels[index])
print(len(mndata.train_images))
print(len(mndata.test_images))
# print(dir(mndata))

annealing = 600.0

def cap(this_num):
    if this_num > 1.0:
        return 1.0
    if this_num < -1.0:
        return -1.0
    return this_num

class Brain:
    def __init__(self):
        self.input_layer = Layer(256)
        self.hidden_layer1 = Layer(144)
        #self.hidden_layer2 = Layer(64)
        self.hidden_layer3 = Layer(16)
        self.output_layer = Layer(10)
        self.input_layer.attach(self.hidden_layer1, [16,12])
        # self.hidden_layer1.attach(self.hidden_layer2, [12,8])
        # self.hidden_layer2.attach(self.hidden_layer3, "f")
        self.hidden_layer1.attach(self.hidden_layer3, "f")
        self.hidden_layer3.attach(self.output_layer, "f")
    def process(self, image):
        self.input_layer.process(image)
        self.hidden_layer1.process("none")
        # self.hidden_layer2.process("none")
        self.hidden_layer3.process("none")
        self.output_layer.process("none")
    def learn(self, label):
        target_list = [-1.0] * 10
        target_list[label] = 1.0
        self.output_layer.calc_error(target_list)
        self.hidden_layer3.calc_error("none")
        # self.hidden_layer2.calc_error("none")
        self.hidden_layer1.calc_error("none")
    annealing += 1.0/150.0

class Layer:
    def __init__(self, count):
        self.neurons = []
        i = 0
        while i < count:
            self.neurons.append(Neuron())
            i += 1
    def attach(self, target_layer, convoluted):
        for idx1, self_neuron in enumerate(self.neurons):
            for idx2, target_neuron in enumerate(target_layer.neurons):
                if convoluted != "f":
                    idx1_y = idx1 / convoluted[0]
                    idx1_x = idx1 % convoluted[0]
                    idx2_y = idx2 / convoluted[1]
                    idx2_x = idx2 % convoluted[1]
                    if idx1_y >= idx2_y and idx1_y < idx2_y + 5:
                        if idx1_x >= idx2_x and idx1_x < idx2_x + 5:
                            self_neuron.synapse_onto(target_neuron)
                else:
                    self_neuron.synapse_onto(target_neuron)
    def process(self, image):
        for idx, neuron in enumerate(self.neurons):
            if image == "none":
                neuron.receive_input("none")
            else:
                neuron.receive_input(image[idx])
    def render(self):
        ary = []
        for idx, neuron in enumerate(self.neurons):
            ary.append(neuron.activity)
        return ary
    def calc_error(self, targets):
        for idx, neuron in enumerate(self.neurons):
            if targets == "none":
                target = neuron.target
            else:
                neuron.target = targets[idx]
            neuron.calc_error()

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
            for idx, input_synapse in enumerate(self.inputs):
                add_up = add_up + input_synapse.get_act() * constant_factor
            add_up = cap(add_up)
            self.activity = add_up
        else:
            self.activity = (brightness/256.0)
        self.target = self.activity ##reset target weight when feeding forward
    def synapse_onto(self, neuron):
        syn = Synapse(self, neuron)
        self.outputs.append(syn)
        neuron.inputs.append(syn)
    def calc_error(self):
        self.target = cap(self.target)
        err = self.activity - self.target
        if err < 0.0:
            err = err * err * -1
        else:
            err = err * err
        delta = annealing
        for idx, input_synapse in enumerate(self.inputs):
            presynaptic = input_synapse.presynaptic
            effect = presynaptic.activity
            presynaptic.target = presynaptic.target - (input_synapse.weight * err / (0.5 * delta * len(presynaptic.outputs)))
            input_synapse.change_weight(err * effect * -1, delta)

class Synapse:
    def __init__(self, neuron1, neuron2):
        self.weight = random() * 2.0 - 1.0
        self.presynaptic = neuron1
        self.postsynaptic = neuron2
        neuron1.outputs.append(self)
        neuron2.inputs.append(self)
    def get_act(self):
        return self.weight * self.presynaptic.activity
    def change_weight(self, amount, delta):
        self.weight = self.weight + (amount/delta)
        self.weight = cap(self.weight)

def evaluate(x, num, mndata):
    j = num-1
    correct = 0
    incorrect = 0
    while j >= 0:
        x.process(mndata.test_images[j])
        output = x.output_layer.render()
        estimate = output.index(max(output))
        truth = mndata.test_labels[j]
        if estimate == truth:
            correct += 1.0
        else:
            incorrect += 1.0
        j -= 1
    print(incorrect/num)
    print(correct/num)

x = Brain()
i = 59999
while i >= 0:
    if i % 1000 == 0:
        print(i)
    if i % 10000 == 0:
        evaluate(x, 1000, mndata)
    x.process(mndata.train_images[i])
    x.learn(mndata.train_labels[i])
    i -= 1

print("Training half done")

while i < 60000:
    if i % 10000 == 0:
        print(i)
        evaluate(x, 1000, mndata)
    x.process(mndata.train_images[i])
    x.learn(mndata.train_labels[i])
    i += 1

correct = 0
incorrect = 0
j = 9999
while j >= 0:
    x.process(mndata.test_images[j])
    output = x.output_layer.render()
    estimate = output.index(max(output))
    truth = mndata.test_labels[j]
    print (estimate, truth)
    if estimate == truth:
        correct += 1.0
        print(output)
    else:
        incorrect += 1.0
        print(output)
    x.learn(mndata.test_labels[j])
    j -= 1

print(incorrect/10000.0)
print(correct/10000.0)


#
