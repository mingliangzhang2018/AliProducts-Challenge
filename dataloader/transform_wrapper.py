# from utils.registry import Registry
import torchvision.transforms as transforms

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn


TRANSFORMS = Registry()


@TRANSFORMS.register("random_resized_crop")
def random_resized_crop(cfg, **kwargs):
    size = kwargs[
        "input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.RandomResizedCrop(
        size=size,
        scale=(0.08,
               1.0),  #cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.SCALE,
        ratio=(0.75, 1.333333333
               )  #cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.RATIO,
    )


@TRANSFORMS.register("random_crop")
def random_crop(cfg, **kwargs):
    size = kwargs[
        "input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.RandomCrop(
        size,
        padding=4  #cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_CROP.PADDING
    )


@TRANSFORMS.register("random_horizontal_flip")
def random_horizontal_flip(cfg, **kwargs):
    return transforms.RandomHorizontalFlip(p=0.5)


@TRANSFORMS.register("shorter_resize_for_crop")
def shorter_resize_for_crop(cfg, **kwargs):
    size = kwargs[
        "input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    assert size[0] == size[1], "this img-process only process square-image"
    return transforms.Resize(int(size[0] / 0.875))


@TRANSFORMS.register("normal_resize")
def normal_resize(cfg, **kwargs):
    size = kwargs[
        "input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.Resize(size)


@TRANSFORMS.register("center_crop")
def center_crop(cfg, **kwargs):
    size = kwargs[
        "input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.CenterCrop(size)


@TRANSFORMS.register("ten_crop")
def ten_crop(cfg, **kwargs):
    size = kwargs[
        "input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.TenCrop(size)


@TRANSFORMS.register("normalize")
def normalize(cfg, **kwargs):
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
