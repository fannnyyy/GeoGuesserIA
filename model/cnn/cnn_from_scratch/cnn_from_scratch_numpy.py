import json
import numpy as np

class Conv:
    def __init__(self, in_channels, out_channels):
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.filters = np.random.randn(out_channels, 3, 3, in_channels)/9

    def iterate_regions(self, image):
        h, w, _ = image.shape 

        for i in range(h-2):
            for j in range(w-2):
                    im_region = image[i:(i+3), j:(j+3), :]
                    yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w, _ = input.shape 
        output = np.zeros((h-2, w-2, self.out_channels))

        for im_region, i, j in self.iterate_regions(input):
            for f in range(self.out_channels):
                output[i,j,f] = np.sum(im_region * self.filters[f])
        return output

    def backprop(self, d_l_d_out, learn_rate):
        d_l_d_filters = np.zeros(self.filters.shape)
        d_l_d_input = np.zeros(self.last_input.shape) 
        for im_region, i,j in self.iterate_regions(self.last_input):
            for f in range(self.out_channels):
                d_l_d_filters[f] += d_l_d_out[i,j,f] * im_region
                d_l_d_input[i:(i+3), j:(j+3), :] += d_l_d_out[i, j, f] * self.filters[f]


        self.filters -= learn_rate * d_l_d_filters

        return d_l_d_input


class MaxPool:
    def iterate_regions(self, image):
        h,w, _ = image.shape 
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region,i,j

    def forward(self, input):
        self.last_input = input
        h,w, out_channels = input.shape
        output = np.zeros((h//2, w//2, out_channels))

        for im_region, i ,j in self.iterate_regions(input):
            output[i,j] = np.amax(im_region, axis=(0,1))

        return output

    def backprop(self, d_l_d_out):
        d_l_d_inputs = np.zeros(self.last_input.shape)

        for im_region,i,j in self.iterate_regions(self.last_input):
            h,w,f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if (im_region[i2,j2,f2] == amax[f2]):
                            d_l_d_inputs[i*2+i2, j*2+j2, f2] = d_l_d_out[i,j,f2]
                            break
        return d_l_d_inputs


class Softmax:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes)/input_len
        self.biases = np.zeros(nodes)

    def forward(self, input):
        self.last_input_shape = input.shape
        input = input.flatten()
        self.last_input = input

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return (exp/np.sum(exp, axis = 0))



    def backprop(self, d_l_d_out, learn_rate):
        d_l_d_inputs = None

        for i, gradient in enumerate(d_l_d_out):
            if gradient == 0:
                continue
            
            t_exp = np.exp(self.last_totals)
            S = np.sum(t_exp)

            d_out_d_t = -t_exp[i] * t_exp/(S**2)
            d_out_d_t[i] = t_exp[i] * (S-t_exp[i])/(S**2)

            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            d_l_d_t = gradient * d_out_d_t

            d_l_d_w = d_t_d_w[np.newaxis].T @ d_l_d_t[np.newaxis]
            d_l_d_b = d_l_d_t * d_t_d_b
            d_l_d_inputs = d_t_d_inputs @ d_l_d_t

            self.weights -= learn_rate * d_l_d_w
            self.biases -= learn_rate * d_l_d_b

        if d_l_d_inputs is not None:
            return d_l_d_inputs.reshape(self.last_input_shape)
        else:
            return np.zeros(self.last_input_shape)

class ReLU:
    def forward(self, input):
        self.last_input = input
        return np.maximum(0, input)
    
    def backprop(self, d_l_d_out):
        return d_l_d_out * (self.last_input > 0)

import json
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
from collections import Counter

csv_path = "/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/metadata_filtered/samples_filtered.csv"
root_dir = "/usr/users/geoguessr_ia/badoul_fan/GeoGuesserIA/dataset_OSV5M/datasets/osv5m/images/samples"

subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
train_dirs = subdirs[:4]
test_dirs = subdirs[4:] 

df = pd.read_csv(csv_path)

unique_labels = sorted(df['country'].unique())
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
num_classes = len(unique_labels)

print("unique_labels : ", unique_labels[0])
print("Exemple label_to_idx : ", list(label_to_idx.items())[:3])
print("Exemple idx_to_label : ", list(idx_to_label.items())[:3])
print("num_classes : ", num_classes)


def load_dataset(folders, root_dir, label_to_idx):
    images = []
    labels = []
    errors = 0
    total_files = 0

    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        files = os.listdir(folder_path)
        
        for filename in files:
            if not filename.endswith(('.jpg', '.png', '.jpeg')):
                continue
            
            img_path = os.path.join(folder_path, filename)
            
            try:
                img = Image.open(img_path)
                img_gray = img.convert("L")
                img_resized = img_gray.resize((64, 64))
                img_array = np.array(img_resized, dtype=np.float32)
                
                img_id = int(filename.split('.')[0])
                
                matching_rows = df[df['id'] == img_id]
                if len(matching_rows) > 0:
                    country = matching_rows.iloc[0]['country']
                    
                    images.append(img_array)
                    labels.append(label_to_idx[country])
                
            except Exception as e:
                errors += 1
                continue
    
    images = np.array(images)
    labels = np.array(labels)
    images = images[..., np.newaxis]
    
    return images, labels

train_images, train_labels = load_dataset(train_dirs, root_dir, label_to_idx)
test_images, test_labels = load_dataset(test_dirs, root_dir, label_to_idx)

conv8 = Conv(in_channels=1, out_channels=8)
relu1 = ReLU()
pool1 = MaxPool()
conv16 = Conv(in_channels=8, out_channels=16)
relu2 = ReLU()
pool2 = MaxPool()
conv32 = Conv(in_channels=16, out_channels=32)
relu3 = ReLU()
pool3 = MaxPool()

softmax = Softmax(6 * 6 * 32, num_classes)

def forward(image, label):
    out = conv8.forward((image/255) - 0.5)
    out = relu1.forward(out)
    out = pool1.forward(out)

    out = conv16.forward(out)
    out = relu2.forward(out)
    out = pool2.forward(out)

    out = conv32.forward(out)
    out = relu3.forward(out)
    out = pool3.forward(out)

    out = softmax.forward(out)

    loss = -np.log(out[label])
    acc = 1 if(np.argmax(out) == label) else 0

    return out, loss, acc

def train(im, label, lr=0.005):
    out,loss,acc = forward(im, label)

    gradient = np.zeros(num_classes)
    gradient[label] = -1/out[label]

    gradient = softmax.backprop(gradient,lr)

    gradient = pool3.backprop(gradient)
    gradient = relu3.backprop(gradient)
    gradient = conv32.backprop(gradient, lr)

    gradient = pool2.backprop(gradient)
    gradient = relu2.backprop(gradient)
    gradient = conv16.backprop(gradient, lr)

    gradient = pool1.backprop(gradient)
    gradient = relu1.backprop(gradient)
    gradient = conv8.backprop(gradient, lr)

    return loss, acc


for epoch in range(3):
    print('----EPOCH %d ---'%(epoch+1))

    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    loss = 0
    num_correct = 0

    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if(i>0 and i % 100 == 99):
            print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))

            loss = 0
            num_correct = 0
        l, acc = train(im,label)
        loss += l
        num_correct += acc

test_loss = 0
test_correct = 0

for im, label in zip(test_images, test_labels):
    out, loss, acc = forward(im, label)
    test_loss += loss
    test_correct += acc

test_accuracy = (test_correct / len(test_images)) * 100
avg_test_loss = test_loss / len(test_images)

print(f"\nTest Accuracy: {test_accuracy:.2f}%")
print(f"Average Test Loss: {avg_test_loss:.3f}")
print(f"Correct predictions: {test_correct}/{len(test_images)}")