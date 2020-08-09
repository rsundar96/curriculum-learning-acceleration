########################################################################
#
# This file is part of the TensorFlow Tutorials available at:
#
# https://github.com/Hvass-Labs/TensorFlow-Tutorials
#
# Published under the MIT License. See the file LICENSE for details.
#
# Copyright 2016 by Magnus Erik Hvass Pedersen
#
########################################################################

import os
import pickle
import numpy as n


def cache(cache_path, fn, *args, **kwargs):

    # If the cache-file exists.
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.

        # Call the function / class-init with the supplied arguments.
        obj = fn(*args, **kwargs)

        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj


def convert_numpy2pickle(in_path, out_path):

    # Load the data using numpy.
    data = np.load(in_path)

    # Save the data using pickle.
    with open(out_path, mode='wb') as file:
        pickle.dump(data, file)


if __name__ == '__main__':
    # This is a short example of using a cache-file.

    # This is the function that will only get called if the result
    # is not already saved in the cache-file. This would normally
    # be a function that takes a long time to compute, or if you
    # need persistent data for some other reason.
    def expensive_function(a, b):
        return a * b

    print('Computing expensive_function() ...')

    # Either load the result from a cache-file if it already exists,
    # otherwise calculate expensive_function(a=123, b=456) and
    # save the result to the cache-file for next time.
    result = cache(cache_path='cache_expensive_function.pkl',
                   fn=expensive_function, a=123, b=456)

    print('result =', result)

    # Newline.
    print()

    # This is another example which saves an object to a cache-file.

    # We want to cache an object-instance of this class.
    # The motivation is to do an expensive computation only once,
    # or if we need to persist the data for some other reason.
    class ExpensiveClass:
        def __init__(self, c, d):
            self.c = c
            self.d = d
            self.result = c * d

        def print_result(self):
            print('c =', self.c)
            print('d =', self.d)
            print('result = c * d =', self.result)

    print('Creating object from ExpensiveClass() ...')

    # Either load the object from a cache-file if it already exists,
    # otherwise make an object-instance ExpensiveClass(c=123, d=456)
    # and save the object to the cache-file for the next time.
    obj = cache(cache_path='cache_ExpensiveClass.pkl',
                fn=ExpensiveClass, c=123, d=456)

    obj.print_result()
