from __future__ import print_function

import os
from collections import defaultdict
import cPickle as pickle
import matplotlib.pyplot as plt

import keras.backend as K
from keras.datasets import mnist
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.initializers import *
from keras.callbacks import *
from keras.utils.generic_utils import Progbar
from keras.preprocessing.image import ImageDataGenerator
import scipy.io

RND = 666

RUN = 'F'
WTS = 'W'
OUT_DIR = 'out/' + RUN
OUT_DIR_W = 'out/' + WTS

# GPU # 
GPU = "1"
# latent vector size
Z_SIZE = 100
# number of iterations D is trained for per each G iteration
D_ITERS = 5
ITERATIONS = 50000
BATCH_SIZE = 100
img_rows = 860
img_cols = 400
save_step = 2000

np.random.seed(RND)

if not os.path.isdir(OUT_DIR): os.makedirs(OUT_DIR)
if not os.path.isdir(OUT_DIR_W): os.makedirs(OUT_DIR_W)
# use specific GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

K.set_image_dim_ordering('tf')
# basically return mean(y_pred),
# but with ability to inverse it for minimization (when y_true == -1)
def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)
    
def create_D():

    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0., stddev=0.02)

    input_image = Input(shape=(img_rows, img_cols, 1), name='input_image')

    x = Conv2D(
        16, (3, 3),
        padding='same',
        name='conv_1',
        kernel_initializer=weight_init)(input_image)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        32, (3, 3),
        padding='same',
        name='conv_2',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        64, (3, 3),
        padding='same',
        name='conv_3',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=2)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(
        128, (3, 3),
        padding='same',
        name='conv_4',
        kernel_initializer=weight_init)(x)
    x = MaxPool2D(pool_size=1)(x)
    x = LeakyReLU()(x)
    x = Dropout(0.3)(x)

    features = Flatten()(x)

    output_is_fake = Dense(1, activation='linear', name='output_is_fake')(features)
    output_class = Dense(1, activation='softmax', name='output_class')(features)

    return Model(
        inputs=[input_image], outputs=[output_is_fake, output_class], name='D')
        
        
def create_G(Z_SIZE=Z_SIZE):
    DICT_LEN = 4
    EMBEDDING_LEN = Z_SIZE

    # weights are initlaized from normal distribution with below params
    weight_init = RandomNormal(mean=0., stddev=0.02)

    # class#
    input_class = Input(shape=(1, ), dtype='int32', name='input_class')
    e = Embedding(
        DICT_LEN, EMBEDDING_LEN,
        embeddings_initializer='glorot_uniform')(input_class)
    embedded_class = Flatten(name='embedded_class')(e)

    # latent var
    input_z = Input(shape=(Z_SIZE, ), name='input_z')

    # hadamard product
    h = multiply([input_z, embedded_class], name='h')

    # cnn part
    x = Dense(256)(h)
    x = LeakyReLU()(x)
	
	x = Dense(2048)(h)
    x = LeakyReLU()(x)

    x = Dense(32 * (img_rows/4) * (img_cols/4))(x)
    x = LeakyReLU()(x)
    x = Reshape(((img_rows/4), (img_cols/4), 16))(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(64, (5, 5), padding='same', kernel_initializer=weight_init)(x)
    x = LeakyReLU()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(32, (5, 5), padding='same', kernel_initializer=weight_init)(x)
    x = LeakyReLU()(x)

    x = Conv2D(
        1, (2, 2),
        padding='same',
        activation='tanh',
        name='output_generated_image',
        kernel_initializer=weight_init)(x)

    return Model(inputs=[input_z, input_class], outputs=x, name='G')


D = create_D()
D.compile(
    optimizer=RMSprop(lr=0.00005),
    loss=[wasserstein, 'sparse_categorical_crossentropy'])
    

input_z = Input(shape=(Z_SIZE, ), name='input_z_')
input_class = Input(shape=(1, ),name='input_class_', dtype='int32')

G = create_G()
# create combined D(G) model
output_is_fake, output_class = D(G(inputs=[input_z, input_class]))
DG = Model(inputs=[input_z, input_class], outputs=[output_is_fake, output_class])
DG.get_layer('D').trainable = False # freeze D in generator training faze

DG.compile(
    optimizer=RMSprop(lr=0.00005),
    loss=[wasserstein, 'sparse_categorical_crossentropy']
)

# #################################################################################################
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory() # build generator as input to the GAN model
# #################################################################################################

train_history = defaultdict(list)

samples_zz = np.random.normal(0., 1., (100, Z_SIZE))
def generate_samples(n=0, save=True):

    generated_classes = np.array(list(range(0, 1)) * 100)
    generated_images = G.predict([samples_zz, generated_classes.reshape(-1, 1)])
    
    
    rr = []
    for c in range(10):
        rr.append(
            np.concatenate(generated_images[c * 10:(1 + c) * 10]).reshape(
                (img_rows * 10), img_cols))
    img = np.hstack(rr)
    

    if save:
        plt.imsave(OUT_DIR + '/samples_%07d.png' % n, img, cmap=plt.cm.gray)
        scipy.io.savemat(OUT_DIR + '/generated_%07d.mat' % n, mdict={'img': img})
        
    return img


def update_tb_summary(step, sample_images=True, save_image_files=True):
    # generated image
    if sample_images:
        img = generate_samples(step, save=save_image_files)
        



progress_bar = Progbar(target=ITERATIONS)

DG_losses = []
D_true_losses = []
D_fake_losses = []

for it in range(ITERATIONS):

    print('Epoch {} of {}'.format(it, ITERATIONS))
    if len(D_true_losses) > 0:
        progress_bar.update(
            it,
            values=[
                    ('D_real_is_fake', np.mean(D_true_losses[-5:], axis=0)[1]),
                    ('D_real_class', np.mean(D_true_losses[-5:], axis=0)[2]),
                    ('D_fake_is_fake', np.mean(D_fake_losses[-5:], axis=0)[1]),
                    ('D_fake_class', np.mean(D_fake_losses[-5:], axis=0)[2]),
                    ('D(G)_is_fake', np.mean(DG_losses[-5:],axis=0)[1]),
                    ('D(G)_class', np.mean(DG_losses[-5:],axis=0)[2])
            ]
        )
        
    else:
        progress_bar.update(it)

    # 1: train D on real+generated images

    if (it % 1000) < 25 or it % 500 == 0: # 25 times in 1000, every 500th
        d_iters = 500
    else:
        d_iters = D_ITERS

    for d_it in range(d_iters):

        # unfreeze D
        D.trainable = True
        for l in D.layers: l.trainable = True
        
        for l in D.layers:
            weights = l.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            l.set_weights(weights)

        
        for real_images, real_images_classes in train_generator:
            break
        
        D_loss = D.train_on_batch(real_images, [-np.ones(BATCH_SIZE), real_images_classes])
        D_true_losses.append(D_loss)

        zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE))
        generated_classes = np.random.randint(0, 1, BATCH_SIZE)
        generated_images = G.predict([zz, generated_classes.reshape(-1, 1)])

        D_loss = D.train_on_batch(generated_images, [np.ones(BATCH_SIZE), generated_classes])
        D_fake_losses.append(D_loss)

    D.trainable = False
    for l in D.layers: l.trainable = False

    zz = np.random.normal(0., 1., (BATCH_SIZE, Z_SIZE))
    generated_classes = np.random.randint(0, 1, BATCH_SIZE)

    DG_loss = DG.train_on_batch(
        [zz, generated_classes.reshape((-1, 1))],
        [-np.ones(BATCH_SIZE), generated_classes])

    DG_losses.append(DG_loss)

    # generate an epoch report on performance
    train_history['DG_loss'].append(DG_loss)
    train_history['D_loss'].append(-D_true_losses[-1][1] - D_fake_losses[-1][1])
    
    if it == 0:
        update_tb_summary(it, sample_images=(it == 0), save_image_files=True)
    
    
    if (it + 1) % save_step == 0:
        update_tb_summary(it, sample_images=((it + 1) % save_step == 0), save_image_files=True)
        
        
    if (it + 1) % save_step == 0:
        G.save(OUT_DIR_W + '/model_G_epoch_{0:03d}.hdf5'.format(it + 1))
        D.save(OUT_DIR_W + '/model_D_epoch_{0:03d}.hdf5'.format(it + 1))
    
    
pickle.dump({'train': train_history}, open('wacgan-history.pkl', 'wb'))



















