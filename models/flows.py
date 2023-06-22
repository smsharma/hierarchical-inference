from functools import partial

import torch
from torch import nn, tanh, relu

from nflows import distributions as distributions_
from nflows import flows, transforms
from nflows.nn import nets


def build_maf(dim=1, num_transforms=8, context_features=None, hidden_features=128):
    transform = transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=dim,
                        hidden_features=hidden_features,
                        context_features=context_features,
                        num_blocks=2,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=tanh,
                        dropout_probability=0.0,
                        use_batch_norm=False,
                    ),
                    transforms.RandomPermutation(features=dim),
                ]
            )
            for _ in range(num_transforms)
        ]
    )

    distribution = distributions_.StandardNormal((dim,))
    neural_net = flows.Flow(transform, distribution)

    return neural_net


def create_alternating_binary_mask(features, even=True):
    """
    Creates a binary mask of a given dimension which alternates its masking.
    :param features: Dimension of mask.
    :param even: If True, even values are assigned 1s, odd 0s. If False, vice versa.
    :return: Alternating binary mask of type torch.Tensor.
    """
    mask = torch.zeros(features).byte()
    start = 0 if even else 1
    mask[start::2] += 1
    return mask


def mask_in_layer(i, features):
    return create_alternating_binary_mask(features=features, even=(i % 2 == 0))


def build_nsf(dim=1, num_transforms=8, context_features=None, hidden_features=128, tail_bound=3.0, num_bins=10):
    conditioner = partial(
        nets.ResidualNet,
        hidden_features=hidden_features,
        context_features=context_features,
        num_blocks=2,
        activation=relu,
        dropout_probability=0.0,
        use_batch_norm=False,
    )

    # Stack spline transforms.
    transform_list = []
    for i in range(num_transforms):
        block = [
            transforms.PiecewiseRationalQuadraticCouplingTransform(
                mask=mask_in_layer(i, dim),
                transform_net_create_fn=conditioner,
                num_bins=num_bins,
                tails="linear",
                tail_bound=tail_bound,
                apply_unconditional_transform=False,
            )
        ]

        # Add LU transform only for high D x. Permutation makes sense only for more than
        # one feature.
        block.append(
            transforms.LULinear(dim, identity_init=True),
        )
        transform_list += block

    distribution = distributions_.StandardNormal((dim,))

    # Combine transforms.
    transform = transforms.CompositeTransform(transform_list)
    neural_net = flows.Flow(transform, distribution)

    return neural_net


def build_mlp(input_dim, hidden_dim, output_dim, layers):
    """Create a MLP from the configurations"""

    activation = nn.GELU

    seq = [nn.Linear(input_dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)
