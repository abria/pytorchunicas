import numpy as np
import yaml

class Preprocessing(object):
    """
    An abstract class representing a preprocessing method.

    All other preprocessing classes should subclass it. All subclasses should override
     - ``__fit__``, that computes the parameters (if any) from a set of images
     - ``apply``, that applies the preprocessing to a given image using the estimated parameters (if any)
     - ``load``, that loads the parameters (if any) from the given path
     - ``save``, that saves the parameters (if any) to the given path
    """

    def fit(self, images):
        raise NotImplementedError

    def apply(self, image):
        raise NotImplementedError

    def revert(self, image):
        raise NotImplementedError

    def load(self, path):
        raise NotImplementedError

    def save(self, path):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class NoPreprocessing(Preprocessing):
    """
    No preprocessing
    """

    def fit(self, images):
        pass

    def apply(self, image):
        return image

    def revert(self, image):
        return image

    def load(self, path):
        pass

    def save(self, path):
        pass

    def name(self):
        return "NoPreprocessing"


class MinMaxNormalization(Preprocessing):
    """
    Min-max normalization preprocessing.

    Data will be normalized to the given range [a,b] w.r.t. to the
    global minimum and maximum values computed from the images passed to 'fit'.
    """

    def __init__(self, a=0.0, b=1.0):
        self.a = a
        self.b = b
        self.min = 0
        self.max = 0

    def fit(self, images):
        self.min = float(np.amin(images))
        self.max = float(np.amax(images))

    def apply(self, image):
        return self.a + (self.b-self.a)*(image - self.min)/(self.max-self.min)

    def revert(self, image):
        return self.min + (self.max-self.min)*(image - self.a)/(self.b-self.a) 

    def load(self, path):
        self.min = np.load(path + '/preprocessing.MinMaxNormalization.min.npy')
        self.max = np.load(path + '/preprocessing.MinMaxNormalization.max.npy')

    def save(self, path):
        np.save(path + '/preprocessing.MinMaxNormalization.min', self.min)
        np.save(path + '/preprocessing.MinMaxNormalization.max', self.max)
        with open(path + '/preprocessing.MinMaxNormalization.min.yaml', 'w') as f:
            yaml.dump(self.min, f)
        with open(path + '/preprocessing.MinMaxNormalization.max.yaml', 'w') as f:
            yaml.dump(self.max, f)

    def name(self):
        return "MinMaxNormalization(" + str(self.a) + ',' + str(self.b) + ')'

class ImageStandardization(Preprocessing):
    """
    Image-wise standardization preprocessing

    Performs image standardization by subtracting the Mean and dividing by the Standard Deviation, for each channel.
	The Mean and the Standard Deviation are calculated from the distribution of all images, channel-wise.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, images):
        self.mean = np.zeros(images[0].shape[2], dtype=np.float64)
        mean_sq = np.zeros(images[0].shape[2], dtype=np.float64)
        for im in images:
            self.mean += np.sum(im, axis=(0, 1), dtype=np.float64)
            mean_sq += np.sum(im.astype(dtype=np.float64, copy=False)**2, axis=(0, 1), dtype=np.float64)
        norm = len(images) * images[0].shape[0] * images[0].shape[1]
        self.mean /= norm
        mean_sq /= norm
        self.std = (mean_sq - self.mean**2)**0.5

    def apply(self, image):
        return (image - self.mean)/self.std

    def revert(self, image):
        return (image * self.std) + self.mean

    def load(self, path):
        self.mean = np.load(path + '/preprocessing.ImageStandardization.mean.npy')
        self.std = np.load(path + '/preprocessing.ImageStandardization.std.npy')

    def save(self, path):
        np.save(path + '/preprocessing.ImageStandardization.mean', self.mean)
        np.save(path + '/preprocessing.ImageStandardization.std', self.std)
        with open(path + '/preprocessing.ImageStandardization.mean.yaml', 'w') as f:
            yaml.dump(self.mean.tolist(), f)
        with open(path + '/preprocessing.ImageStandardization.std.yaml', 'w') as f:
            yaml.dump(self.std.tolist(), f)

    def name(self):
        return "ImageStandardization"

class PixelStandardization(Preprocessing):
    """
    Pixel-wise standardization preprocessing.

    Performs pixel standardization by subtracting the Mean and dividing by the Standard Deviation.
	The Mean and the Standard Deviation are calculated separately for each location, from the distribution 
    of all input patches. Formally, the operation is defined as:
        pixel_output(x,y) = (pixel_input(x,y) - mean(x,y))/stdev(x,y).
	Output values will be approximately normal-distributed at the pixel level.
    This practice is suggested in: 
        LeCun et al., Efficient backprop, Neural networks: Tricks of the trade (1998)
    """

    def __init__(self):
        self.mean = 0
        self.std = 0

    def fit(self, images):
        self.mean = np.zeros(images[0].shape, dtype=np.float64)
        mean_sq = np.zeros(images[0].shape, dtype=np.float64)
        for im in images:
            self.mean += im
            mean_sq += im.astype(dtype=np.float64, copy=False)**2
        self.mean /= len(images)
        mean_sq /= len(images)
        self.std = (mean_sq - self.mean**2)**0.5

    def apply(self, image):
        return (image - self.mean)/self.std

    def revert(self, image):
        return (image * self.std) + self.mean

    def load(self, path):
        self.mean = np.load(path + '/preprocessing.PixelStandardization.mean.npy')
        self.std = np.load(path + '/preprocessing.PixelStandardization.std.npy')

    def save(self, path):
        np.save(path + '/preprocessing.PixelStandardization.mean', self.mean)
        np.save(path + '/preprocessing.PixelStandardization.std', self.std)
        with open(path + '/preprocessing.PixelStandardization.mean.yaml', 'w') as f:
            yaml.dump(self.mean.tolist(), f)
        with open(path + '/preprocessing.PixelStandardization.std.yaml', 'w') as f:
            yaml.dump(self.std.tolist(), f)

    def name(self):
        return 'PixelStandardization'
