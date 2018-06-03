import numpy as np
import os
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, io, color, exposure, img_as_float, transform, img_as_uint
from matplotlib import pyplot as plt

def loadDataGeneral(data_path, image_list, im_shape):
    X = []
    image_dir = os.path.join(data_path, 'images')
    image_names = []

    with open(image_list, 'r') as f:
    	for line in f:
    		image_file = os.path.join(image_dir, line[:-1])
    		img = img_as_float(io.imread(image_file))
    		if len(img.shape) == 3:
    			continue
    		img = transform.resize(img, im_shape)
    		img = exposure.equalize_hist(img)
    		img = np.expand_dims(img, -1)
    		
    		X.append(img)
    		image_names.append(image_file)

    X = np.array(X)
    #print(X.shape)
    X -= X.mean()
    X /= X.std()

    print '### Dataset loaded'
    print 'path = {}'.format(data_path)
    print 'data shape: {}'.format(X.shape)
    print 'min and max value, X:{:.1f}, {:.1f}'.format(X.min(), X.max())
    print 'statistic, X.mean = {}, X.std = {}'.format(X.mean(), X.std())
    return X, image_names

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

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    data_path = '/Users/chaoyan/Documents/Nvidia/CXR/'
    image_list = os.path.join(data_path, 'list.txt')

    # Load test data
    im_shape = (256, 256)
    X, image_names = loadDataGeneral(data_path, image_list, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_name = '../trained_model.hdf5'
    UNet = load_model(model_name)

    # For inference standard keras ImageGenerator can be used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    gts, prs = [], []
    i = 0
    for xx in test_gen.flow(X, shuffle=False, batch_size=1):
  		img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
  		pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        #mask = yy[..., 0].reshape(inp_shape[:2])

        #gt = mask > 0.5
		pr = pred > 0.5
		pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
		
		io.imsave(image_names[i] + '.mask.jpg',  img_as_uint(pr))

		"""
		plt.imshow(pr, 'gray')
		plt.axis('off')
		plt.savefig(image_names[i] + '.mask.jpg', bbox_inches='tight')
		#plt.show()
		plt.close()
		"""
		i += 1

		if n_test == i:
			break


