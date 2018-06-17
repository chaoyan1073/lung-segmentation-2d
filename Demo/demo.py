import numpy as np
import os
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform, img_as_uint
from matplotlib import pyplot as plt

def loadDataGeneral(load_img_names, im_shape):
    X = []
    #print '---------------------start read data'
    cnt = 0
    for image_file in load_img_names:
        img = img_as_float(io.imread(image_file))
        if len(img.shape) == 3:
            print '---------------------wrong size:', image_file
            continue
        img = transform.resize(img, im_shape)
		#img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        X.append(img)

    X = np.array(X)
    #print '---------------------X shape:', X.shape
    X -= X.mean()
    X /= X.std()

    #print '### Dataset loaded'
    print 'path = {}'.format(data_path)
    #print 'data shape: {}'.format(X.shape)
    #print 'min and max value, X:{:.1f}, {:.1f}'.format(X.min(), X.max())
    #print 'statistic, X.mean = {}, X.std = {}'.format(X.mean(), X.std())
    return X, load_img_names

def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    data_path = '/home/allen/Documents/Data/PLCO_rotated'
    image_list = '/home/allen/Documents/Data/list.txt'

    image_names = []
    with open(image_list, 'r') as f:
        for line in f:
            items = line.split()
            image_name = os.path.join(data_path, items[0])
            image_names.append(image_name)
    
    image_name_lists = partition(image_names, 100)

    # Load test data
    im_shape = (256, 256)
    # Load model
    model_name = '../trained_model.hdf5'
    UNet = load_model(model_name)
    for name_list in image_name_lists:
        print 'load images number:', len(name_list)
        X, img_names = loadDataGeneral(name_list, im_shape)
        n_test = X.shape[0]
        inp_shape = X[0].shape

        # For inference standard keras ImageGenerator can be used.
        test_gen = ImageDataGenerator(rescale=1.)

        batches = 0
        bs = 64
        for xx, yy in test_gen.flow(X, img_names, shuffle=False, batch_size=bs):
      		#img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
      		#pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
            #mask = yy[..., 0].reshape(inp_shape[:2])
            pred = UNet.predict(xx)
            for idx in range(bs):
                prediction = pred[idx, ..., 0]
                pr = prediction > 0.5
                pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
                na, et = os.path.splitext(yy[idx])
                na = na + '.jpg'
                io.imsave(na,  img_as_uint(pr))
                """
                plt.imshow(pr, 'gray')
                plt.axis('off')
                plt.savefig(image_names[i] + '.mask.jpg', bbox_inches='tight')
                #plt.show()
                plt.close()
                """
            batches += 1
            if batches >= n_test / bs:
                break


