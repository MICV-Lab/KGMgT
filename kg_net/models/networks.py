from kg_net.models.archs import _arch_modules
import pdb


def dynamical_instantiation(modules, cls_type, opt):
    """Dynamically instantiate class.

    Args:
        modules (list[importlib modules]): List of modules from importlib
        files.
        cls_type (str): Class type.
        opt (dict): Class initialization kwargs.

    Returns:
        class： Instantiated class.
    """
    
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:

        raise ValueError(f'{cls_} is not found.')

    return cls_(**opt)


# generator
# opt：{name：test_restiration_mse,suffix:None,scale:4......}
def define_net_g(opt):

    opt_net = opt['network_g']
    network_type = opt_net.pop('type')
    net_g = dynamical_instantiation(_arch_modules, network_type, opt_net)

    return net_g


# Discriminator
def define_net_d(opt):
    opt_net = opt['network_d']
    network_type = opt_net.pop('type')

    net_d = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_d

def define_net_ae(opt):
    opt_net = opt['network_ae']
    network_type = opt_net.pop('type')

    net_ae = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_ae

def define_net_refine(opt):
    opt_net = opt['network_refine']
    network_type = opt_net.pop('type')

    net_refine = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_refine

def define_net_noStudent_map(opt):
    opt_net = opt['network_noStudent_map']
    network_type = opt_net.pop('type')

    net_noStudent_map = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_noStudent_map

def define_net_map(opt):
    opt_net = opt['network_map']
    network_type = opt_net.pop('type')

    net_map = dynamical_instantiation(_arch_modules, network_type, opt_net)
    return net_map

def define_net_extractor(opt):
    opt_net = opt['network_extractor']
    network_type = opt_net.pop('type')

    net_extractor = dynamical_instantiation(_arch_modules, network_type,
                                            opt_net)
    return net_extractor


def define_net_student(opt):
    opt_net = opt['network_student']
    network_type = opt_net.pop('type')

    net_student = dynamical_instantiation(_arch_modules, network_type, opt_net)

    return net_student


def define_net_teacher(opt):
    opt_net = opt['network_teacher']
    network_type = opt_net.pop('type')

    net_teacher = dynamical_instantiation(_arch_modules, network_type, opt_net)

    return net_teacher
