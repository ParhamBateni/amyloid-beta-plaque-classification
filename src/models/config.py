class Config:
    def __init__(self, config: dict):
        self.config = {k: Config(v) if isinstance(v, dict) else v for k, v in config.items()}

    def __getattr__(self, name: str):
        if name not in self.config:
            raise AttributeError(f"Config has no attribute {name}")
        return self.config[name]

    def _indented_str(self, indent: int = 1):
        return '{\n' + '\t' * indent + (',\n' + '\t' * indent).join([f'{str(k)}: {str(v) if not isinstance(v, Config) else v._indented_str(indent + 1)}' for k, v in self.config.items()]) + '\n' + '\t' * (indent - 1) + '}'

    def __str__(self):
        return self._indented_str()

    def __getitem__(self, key: str):
        return self.config[key]

    def __setitem__(self, key: str, value: any):
        self.config[key] = value

    def __delattr__(self, name: str):
        del self.config[name]

    def to_dict(self):
        """
        Recursively convert Config objects to dictionaries.
        """
        result = {}
        for k, v in self.config.items():
            if isinstance(v, Config):
                result[k] = v.to_dict()
            else:
                result[k] = v
        return result