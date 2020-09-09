import random
import numpy as np

class FlipRotate(object):
    """
    Generates new samples by performing random horizontal and vertical flips 
    and rotations by 0, 90, 180, and 270 degrees.
    """

    def __call__(self, sample):

        # extract random operation
        op = random.randint(0,7)

        if op is 0:
            return sample
        elif op is 1:
            return np.rot90(sample, 1)
        elif op is 2:
            return np.rot90(sample, 2)
        elif op is 3:
            return np.rot90(sample, 3)
        elif op is 4:
            return np.flipud(sample)
        elif op is 5:
            return np.fliplr(sample)
        elif op is 6:
            return np.flipud(np.rot90(sample, 1))
        else:
            return np.fliplr(np.rot90(sample, 1))

class Replicate(object):
    """
    Just replicates the given simple.
    """

    def __call__(self, sample):

        return sample