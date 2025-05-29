import numpy as np
from domainbed.lib import misc


def _hparams(augmentations, dataset, random_seed):
    """
    Global registry of hyperparameters for augmentations.
    Each entry is a (default, random) tuple.
    """
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # AugMix parameters
    if "AugMix" in augmentations:
        _hparam('augmix_severity', 3, lambda r: int(r.uniform(1, 5)))
        _hparam('augmix_all_ops', True, lambda r: bool(r.choice([True, False])))
        _hparam('augmix_mixture_width', 3, lambda r: int(r.uniform(1, 5)))
        _hparam('augmix_chain_depth', -1, lambda r: int(r.uniform(-1, 3)))

    # MixUp parameters
    if "MixUp" in augmentations:
        _hparam('mixup_alpha', 0.2, lambda r: r.uniform(0.1, 0.4))

    # CutMix parameters
    if "CutMix" in augmentations:
        _hparam('cutmix_alpha', 1.0, lambda r: r.uniform(0.5, 1.5))

    # IPMix parameters
    if "IPMix" in augmentations:
        _hparam('ipmix_k', 3, lambda r: int(r.uniform(1, 5)))
        _hparam('ipmix_t', 3, lambda r: int(r.uniform(1, 5)))
        _hparam('ipmix_beta', 4, lambda r: r.uniform(2, 6))
        _hparam('ipmix_severity', 1, lambda r: int(r.uniform(1, 3)))
        _hparam('ipmix_all_ops', True, lambda r: bool(r.choice([True, False])))
        _hparam('ipmix_mixing_set_path', "./domainbed/lib_augmentations/ipmix/ifs",
                lambda r: "./domainbed/lib_ipmix/ifs")

    # Style Transfer parameters
    if "Stylized" in augmentations:
        _hparam('style_dir', "./domainbed/lib_augmentations/stylized/train",
                lambda r: "./domainbed/lib_augmentations/stylized/train")
        _hparam('decoder_pth', "./domainbed/lib_augmentations/stylized/models/decoder.pth",
                lambda r: "./domainbed/lib_augmentations/stylized/models/decoder.pth")
        _hparam('vgg_normalised_pth', "./domainbed/lib_augmentations/stylized/models/vgg_normalised.pth",
                lambda r: "./domainbed/lib_augmentations/stylized/models/vgg_normalised.pth")
        _hparam('content_size', 0, lambda r: int(r.choice([0, 512, 768])))
        _hparam('crop', 0, lambda r: int(r.choice([0, 224])))
        _hparam('style_size', 256, lambda r: int(r.choice([256, 512])))
        _hparam('alpha', 1.0, lambda r: r.uniform(0.5, 1.5))

    # CartoonGAN parameters
    if "CartoonGAN" in augmentations:
        _hparam('cartoongan_model_path', "./domainbed/lib_augmentations/cartoongan/Hayao_net_G_float.pth",
                lambda r: "./domainbed/lib_augmentations/cartoongan/Hayao_net_G_float.pth")

    return hparams


def default_hparams(augmentations, dataset):
    """Get default hyperparameters."""
    return {a: b for a, (b, c) in
            _hparams(augmentations, dataset, 0).items()}


def random_hparams(augmentations, dataset, seed):
    """Get random hyperparameters."""
    return {a: c for a, (b, c) in _hparams(augmentations, dataset, seed).items()}