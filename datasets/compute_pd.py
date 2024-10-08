import numpy as np
import gudhi as gd
from persim import PersImage
import torch
import collections
from itertools import groupby


def compute_pd(points):
    alpha_complex = gd.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()
    diag = simplex_tree.persistence()
    pim = PersImage(spread=1e-2, pixels=[50, 50], verbose=False)

    result = dict()
    for k, itr in groupby(diag, lambda x: x[0]):
        result[k] = [list(v[1]) for v in itr]

    dgm = {}
    for key in result:
        t = np.asarray(result[key])
        t = t[t[:, 1] != np.inf]
        dgm[key] = np.asarray(t)

    if 1 in dgm:
        pis1 = pim.transform(dgm[1]).astype(np.float32)
        pis1 = pis1 / (pis1.max() + 1e-20)
    else:
        pis1 = np.zeros((50, 50))

    if 2 in dgm:
        pis2 = pim.transform(dgm[2]).astype(np.float32)
        pis2 = pis2 / (pis2.max() + 1e-20)
    else:
        pis2 = np.zeros((50, 50))

    pis1 = torch.tensor(pis1.copy()).float()
    pis1 = pis1.view(2500, )

    pis2 = torch.tensor(pis2.copy()).float()
    pis2 = pis2.view(2500, )

    return pis1, pis2


