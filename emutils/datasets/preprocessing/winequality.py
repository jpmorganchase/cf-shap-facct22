import os

import numpy as np
import pandas as pd

from emutils import PACKAGE_DATA_FOLDER
from emutils.utils import attrdict
from ._utils import flip_binary_class

from ..uci import download_uci_dataset


def load_winequality(
    type='white',
    red_filename='winequality-red.csv',
    white_filename='winequality-white.csv',
    binary_target_threshold=None,
):

    dataset_directory = download_uci_dataset('Wine Quality')

    if type == 'white':
        filename = white_filename
    elif type == 'red':
        filename = red_filename
    else:
        raise ValueError('Invalid type. It must be either \'red\' or \'white\'')

    data = pd.read_csv(os.path.join(dataset_directory, filename), sep=';')

    if binary_target_threshold is not None:
        data['quality'] = data['quality'].apply(lambda x: (x < binary_target_threshold) * 1)
        class_names = ['Bad', 'Good']
    else:
        data['quality'] = data['quality'] - 3
        class_names = [str(x) for x in np.sort(data['quality'].unique())]

    return flip_binary_class(attrdict(
        data=data,
        target_name='quality',
        class_names=class_names,
    ))
