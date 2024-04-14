#!/usr/bin/env python3
""" Flip Me Over """


def matrix_transpose(matrix):
    """ transpose a 2D matrix """
    rows = len(matrix)
    cols = len(matrix[0])

    transpose = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    return transpose
