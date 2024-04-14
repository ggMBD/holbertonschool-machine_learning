#!/usr/bin/env python3
""" Flip Me Over """


def add_arrays(arr1, arr2):
    """ transpose a 2D matrix """
    if (len(arr1)!= len(arr2)):
        return None
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
