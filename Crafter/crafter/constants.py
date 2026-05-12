import pathlib

import ruamel.yaml as yaml

SURVIVAL_LOG = False

root = pathlib.Path(__file__).parent
_yaml = yaml.YAML(typ='safe', pure=True)
for key, value in _yaml.load((root / 'data.yaml').read_text(encoding='utf-8')).items():
    globals()[key] = value
