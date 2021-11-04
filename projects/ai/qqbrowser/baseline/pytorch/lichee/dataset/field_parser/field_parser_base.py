from lichee import config


class BaseFieldParser:
    def __init__(self):
        self.cfg = None
        self.alias = None
        self.key = None
        self.global_config = config.get_cfg()

    def init(self, cfg):
        '''
        :param cfg: field specified config
        :return:
        '''
        self.cfg = cfg

    def set_key(self, alias, key):
        '''
        :param alias: field alias
        :param key: field key
        :return:
        '''
        self.alias = alias
        self.key = key

    def parse(self, record, training=False):
        raise NotImplementedError

    def collate(self, batch):
        raise NotImplementedError
