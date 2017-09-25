import numpy as np
import pandas as pd
import sys
import os
from keras import backend as K
from keras.optimizers import SGD
from skimage.io import imread
from matplotlib import pyplot as plt

from architecture import set_architecture

os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN, device=gpu0, floatX=float32, optimizer=fast_compile'

def label_map(labels):
    label_map = np.zeros([img_h, img_w, n_labels])    
    for r in range(img_h):
        for c in range(img_w):
            label_map[r, c, labels[r][c]] = 1
    return label_map

def prep_data(mode):
    assert mode in {'test', 'train'}, \
        'mode should be either \'test\' or \'train\''
    data = []
    label = []
    df = pd.read_csv(path + mode + '.csv')
    n = n_train if mode == 'train' else n_test
    for i, item in df.iterrows():
        if i >= n:
            break
        img, gt = [imread(path + item[0])], np.clip(imread(path + item[1]), 0, 1)
        data.append(img)
        label.append(label_map(gt))
        sys.stdout.write('\r')
        sys.stdout.write(mode + ": [%-20s] %d%%" % ('=' * int(20. * (i + 1) / n - 1) + '>',
                                                    int(100. * (i + 1) / n)))
        sys.stdout.flush()
    sys.stdout.write('\r')
    sys.stdout.flush()
    data, label = np.array(data), np.array(label).reshape((n, img_h * img_w, n_labels))

    print mode + ': OK'
    print '\tshapes: {}, {}'.format(data.shape, label.shape)
    print '\ttypes:  {}, {}'.format(data.dtype, label.dtype)
    print '\tmemory: {}, {} MB'.format(data.nbytes / 1048576, label.nbytes / 1048576)

    return data, label


def plot_results(output):
    gt = []
    df = pd.read_csv(path + 'test.csv')
    for i, item in df.iterrows():
        gt.append(np.clip(imread(path + item[1]), 0, 1))

    plt.figure(figsize=(15, 2 * n_test))
    for i, item in df.iterrows():
        plt.subplot(n_test, 4, 4 * i + 1)
        plt.title('Ground Truth')
        plt.axis('off')
        gt = imread(path + item[1])
        plt.imshow(np.clip(gt, 0, 1))

        plt.subplot(n_test, 4, 4 * i + 2)
        plt.title('Prediction')
        plt.axis('off')
        labeled = np.argmax(output[i], axis=-1)
        plt.imshow(labeled)

        plt.subplot(n_test, 4, 4 * i + 3)
        plt.title('Heat map')
        plt.axis('off')
        plt.imshow(output[i][:, :, 1])

        plt.subplot(n_test, 4, 4 * i + 4)
        plt.title('Comparison')
        plt.axis('off')
        rgb = np.empty((img_h, img_w, 3))
        rgb[:, :, 0] = labeled
        rgb[:, :, 1] = imread(path + item[0])
        rgb[:, :, 2] = gt
        plt.imshow(rgb)

    plt.savefig('result.png')
    plt.show()


if __name__ == '__main__':
    # inputs 
    path = 'Data/'    # path to data
    img_channels = 1  # img channels
    img_w = 256       # img width (pixels)
    img_h = 256       # img height (pixels)
    n_labels = 2      # number of labels
    n_train = 6       # number of samples in train set
    n_test = 3        # number of samples in test set
    n_epochs = 100    # number of epochs to train
    batch_size = 1    # batch size

    
    # read in data  
    train_data, train_label = prep_data('train')
    test_data, test_label = prep_data('test')

    #if K.image_dim_ordering() == 'th':
    #    X = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    #    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    #    input_shape = (1, img_rows, img_cols)
    #else:
    #    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    #    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    #    input_shape = (img_rows, img_cols, 1)

    input_shape = (img_channels, img_h, img_w) # channels first
    model = set_architecture(n_labels, input_shape)

    #optimizer = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=False)
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    print('Compiled: OK')

    history = model.fit(train_data, train_label, batch_size=batch_size, epochs=n_epochs,
                        validation_data = (test_data, test_label), verbose=1) 
    
    #model.save_weights('weights.hdf5')
    #model.load_weights('model_5l_weight_ep50.hdf5')

    #test_data, test_label = prep_data('test')
    score = model.evaluate(test_data, test_label, verbose=0)
    print 'Test score:', score[0]
    print 'Test accuracy:', score[1]

    output = model.predict(test_data, verbose=0)
    output = output.reshape((output.shape[0], img_h, img_w, n_labels))
    plot_results(output)
