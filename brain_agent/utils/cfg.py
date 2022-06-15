from omegaconf import OmegaConf
from brain_agent.utils.utils import AttrDict


class Configs(OmegaConf):
    @classmethod
    def get_defaults(cls):
        cfg = cls.load('configs/nethack/baseline.yaml')
        cls.set_struct(cfg, True)
        return cfg

    @classmethod
    def override_from_file_name(cls, cfg):
        c = cls.override_from_cli(cfg)
        if not cls.is_missing(c, 'cfg'):
            c = cls.load(c.cfg)
        cfg = cls.merge(cfg, c)
        return cfg

    @classmethod
    def override_from_cli(cls, cfg):
        c = cls.from_cli()
        cfg = cls.merge(cfg, c)
        return cfg

    @classmethod
    def to_attr_dict(cls, cfg):
        c = cls.to_container(cfg)
        c = AttrDict.from_nested_dicts(c)
        return c

