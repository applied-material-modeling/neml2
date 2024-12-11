import math
import torch
import numpy as np

from tqdm import tqdm

import neml2


def gauss_points(deg, device=torch.device("cpu")):
    """Wrap numpy to get Gauss points and weights

    Args:
        deg (int): degree

    Keyword Args:
        device (torch.device): which torch device to use
    """
    pts, wgts = np.polynomial.legendre.leggauss(deg)
    return torch.tensor(pts, device=device), torch.tensor(wgts, device=device)


def spherical_quadrature(deg, device=torch.device("cpu")):
    """Construct a spherical quadature rule (unit sphere)

    Note I'm baking in dV to the weights

    Args:
        deg (int): degree

    Keyword Args:
        device (torch.device): which device to use
    """
    opts, owts = gauss_points(deg, device)
    pts = torch.stack(
        torch.meshgrid(
            (opts + 1.0) / 2.0,
            (opts + 1.0) / 2.0 * np.pi,
            (opts + 1.0) / 2.0 * 2.0 * np.pi,
            indexing="ij",
        ),
        -1,
    ).reshape(-1, 3)
    wts = (
        torch.prod(
            torch.stack(torch.meshgrid(owts, owts, owts, indexing="ij"), -1).reshape(
                -1, 3
            ),
            -1,
        )
        * np.pi**2.0
        / 4.0
        * pts[:, 0] ** 2.0
        * torch.sin(pts[:, 1])
    )

    print(owts)

    return pts, wts


def rotation_quadrature(deg, device=torch.device("cpu")):
    """Construct a rotational quadrature rule

    I am again baking the dV into the weights

    Args:
        deg (int): degree

    Keyword Args:
        device (torch.device): which device to run
    """
    spoints, sweights = spherical_quadrature(deg, device)
    r = spoints[:, 0]
    p = spoints[:, 1]
    t = spoints[:, 2]

    rpoints = torch.stack(
        [
            r * torch.sin(p) * torch.cos(t),
            r * torch.sin(p) * torch.sin(t),
            r * torch.cos(p),
        ],
        -1,
    )
    Rpoints = neml2.tensors.Rot(rpoints)
    Rweights = neml2.tensors.Scalar(sweights) * Rpoints.dV()

    return Rpoints, Rweights


class ODF(torch.nn.Module):
    """Parent class for Orientation Distribution Functions

    Args:
        X (neml2.tensors.Rot): rotations, must have a single batch dimension
    """

    def __init__(self, X):
        super().__init__()
        assert X.batch.dim() == 1
        self.X = X

    @property
    def n(self):
        return self.X.batch.shape[0]

    def texture_index(self, deg=5):
        """Integrate the texture index as a probability

        i.e. int_SO(3) (f(x) / Pi)**2.0 dV

        Keyword Arguments:
            deg (int): quadrature order to use
        """
        Rpoints, Rweights = rotation_quadrature(deg, device=self.X.device)

        vals = (self.forward(Rpoints) / np.pi) ** 2.0

        return torch.sum(vals * Rweights.torch())


def split(X, sf):
    """Helper routine to split a batch of orientations into test/validation sets

    Args:
        X (indexable in first dimension): reference set
        sf (float): split fraction
    """
    l = X.batch.shape[0]
    nv = int(sf * l)
    inds = torch.randperm(l)
    vset = inds[:nv]
    tset = inds[nv:]

    return neml2.tensors.Rot(X.torch()[vset].clone()), neml2.tensors.Rot(
        X.torch()[tset].clone()
    )


class KDEODF(ODF):
    """ODF represented from a Kernel Density Estimate

    Args:
        X (neml2.tensors.Rot): rotations, must have a single batch dimension
        kernel (Kernel): kernel function
    """

    def __init__(self, X, kernel):
        super().__init__(X)
        self.kernel = kernel

    def optimize_kernel(self, miter=250, verbose=False, lr=1.0e-2, sf=0.2):
        """Optimize the kernel half width by maximizing the KL likelihood

        Keyword Args:
            miter (int): optimization iterations
            verbose (bool): if true print convergene progress
            lr (float): learning rate
            sf (float): fraction of data split out for validation
        """
        if verbose:
            it = tqdm(range(miter))
            it.set_description("KL: ")
        else:
            it = range(miter)

        # Make the half width a parameter
        self.kernel.h = torch.nn.Parameter(self.kernel.h)

        # Setup optimizer
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        # Split the data
        X_orig = self.X.clone()

        self.X, val = split(X_orig, sf)

        for _ in it:
            optim.zero_grad()
            KL = -torch.mean(torch.log(self.forward(val) / math.pi))
            if verbose:
                it.set_description("KL: %6.5e" % KL.detach().cpu())
            KL.backward()
            optim.step()

        self.X = X_orig

    def forward(self, Y):
        """Calculate the probability density at each point in Y

        Args:
            Y (neml2.tensors.Rot): rotations with arbitrary batch shape

        Returns:
            torch.tensor with the probabilities
        """
        dist = self.X.dist(Y.batch.unsqueeze(-1)).torch()

        return torch.mean(
            self.kernel(torch.cos(dist / 2.0)),
            dim=-1,
        )


class Kernel(torch.nn.Module):
    """Parent class for kernels for KDE reconstruction

    Args:
        h (torch.tensor): half-width
    """

    def __init__(self, h):
        super().__init__()
        self.h = h


class DeLaValleePoussinKernel(Kernel):
    """De La Vallee Poussin kernel, according to MTEX"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X):
        """Evaluate the kernel

        Args:
            X (torch.tensor):
        """
        kappa = 0.5 * math.log(0.5) / torch.log(torch.cos(self.h / 2))
        c = beta(
            torch.tensor(1.5, device=self.h.device),
            torch.tensor(0.5, device=self.h.device),
        ) / beta(torch.tensor(1.5, device=self.h.device), kappa + 0.5)

        return c * X ** (2 * kappa)


def beta(z1, z2):
    """Calculate the beta function

    Args:
        z1 (torch.tensor): first input
        z2 (torch.tensor): second input
    """
    return torch.exp(
        torch.special.gammaln(z1)
        + torch.special.gammaln(z2)
        - torch.special.gammaln(z1 + z2)
    )
