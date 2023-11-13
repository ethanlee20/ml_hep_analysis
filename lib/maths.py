import numpy as np
import pandas as pd


# Math


def square_matrix_transform(df_matrix, df_vec):
    """
    Multiply a vector dataframe by a square matrix dataframe.

    Vector (matrix) dataframes have one vector per row. The elements of a row are the components of a vector (matrix).

    :param df_matrix: Dataframe of square matrices.
    :param df_vec: Dataframe of vectors.
    :return: Transformed dataframe of vectors.
    """
    assert np.sqrt(df_matrix.shape[1]) == df_vec.shape[1]
    dims = df_vec.shape[1]

    result = pd.DataFrame(
        data=np.zeros(shape=df_vec.shape),
        index=df_vec.index,
        columns=df_vec.columns,
        dtype="float64",
    )

    for i in range(dims):
        for j in range(dims):
            result.iloc[:, i] += df_matrix.iloc[:, dims * i + j] * df_vec.iloc[:, j]
    return result


def dot_product(df_vec1, df_vec2):
    """
    Compute the dot product of vector dataframes.

    Vector dataframes have one vector per row. The elements of a row are the components of a vector.

    :param df_vec1: Dataframe of vectors.
    :param df_vec2: Dataframe of vectors.
    :return: Series of dot products of vectors.
    """
    assert df_vec1.shape[1] == df_vec2.shape[1]
    dims = df_vec1.shape[1]

    result = pd.Series(
        data=np.zeros(len(df_vec1)), index=df_vec1.index, dtype="float64"
    )
    for dim in range(dims):
        result += df_vec1.iloc[:, dim] * df_vec2.iloc[:, dim]
    return result


def vector_magnitude(df_vec):
    """
    Compute the magnitudes of vectors in a dataframe.

    The vector dataframe has one vector per row. The elements of a row are the components of a vector.

    :param df_vec: Dataframe of vectors.
    :return: Series of vector magnitudes.
    """
    return np.sqrt(dot_product(df_vec, df_vec))


def find_cosine_angle(df_vec1, df_vec2):
    """
    Find the cosine of the angle between vector dataframes.

    Given two dataframes of vectors, this function finds the cosine of the angle
    between vectors in the first dataframe and vectors in the second dataframe.
    Vector dataframes have one vector per row. The elements of a row are the components of a vector.

    :param df_vec1: Dataframe of vectors.
    :param df_vec2: Dataframe of vectors.
    :return: Series of cosines.
    """
    return dot_product(df_vec1, df_vec2) / (
            vector_magnitude(df_vec1) * vector_magnitude(df_vec2)
    )


def cross_product_3d(df_3vec1, df_3vec2):
    """
    Find the cross product of two 3-dim vector dataframes.

    Vector dataframes have one vector per row. The elements of a row are the components of a vector.

    :param df_3vec1: Dataframe of 3-dim vectors.
    :param df_3vec2: Dataframe of 3-dim vectors.
    :return: Dataframe of resultant cross product vectors.
    """
    assert df_3vec1.shape[1] == df_3vec2.shape[1] == 3
    assert df_3vec1.shape[0] == df_3vec2.shape[0]
    assert df_3vec1.index.equals(df_3vec2.index)

    def clean(df_3vec):
        df_3vec = df_3vec.copy()
        df_3vec.columns = ["x", "y", "z"]
        return df_3vec

    df_3vec1 = clean(df_3vec1)
    df_3vec2 = clean(df_3vec2)

    result = pd.DataFrame(
        data=np.zeros(shape=df_3vec1.shape),
        index=df_3vec1.index,
        columns=df_3vec1.columns,
        dtype="float64",
    )

    result["x"] = df_3vec1["y"] * df_3vec2["z"] - df_3vec1["z"] * df_3vec2["y"]
    result["y"] = df_3vec1["z"] * df_3vec2["x"] - df_3vec1["x"] * df_3vec2["z"]
    result["z"] = df_3vec1["x"] * df_3vec2["y"] - df_3vec1["y"] * df_3vec2["x"]

    return result


def unit_normal_vector_plane(df_3vec1, df_3vec2):
    """
    Find the unit normal vector dataframe of a 3-dim plane dataframe.

    Vector dataframes have one vector per row. The elements of a row are the components of a vector.

    :param df_3vec1: The first 3-dim vector dataframe that defines the plane dataframe.
    :param df_3vec2: The second 3-dim vector dataframe that defines the plane dataframe.
    :return: Dataframe of unit normal vectors.
    """
    df_normal_vec = cross_product_3d(df_3vec1, df_3vec2)
    df_normal_unit_vec = df_normal_vec.divide(vector_magnitude(df_normal_vec), axis="index")

    return df_normal_unit_vec


def test_unit_normal_vector_plane():
    df = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}, index=[3, 4, 10]
    )
    print(df)
    df1 = pd.DataFrame(
        {"d": [1, 1, 2], "e": [3, 2, 1], "f": [5, 4, 7]}, index=[3, 4, 10]
    )
    print(df1)
    print(unit_normal_vector_plane(df, df1))
