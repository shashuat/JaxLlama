class ConfigModule(hk.Module):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config