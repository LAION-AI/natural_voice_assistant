from omegaconf import DictConfig, OmegaConf, errors as omegaconf_errors
from pytorch_lightning import LightningModule
from conformer_encoder import ConformerEncoder
from rnnt_decoder import RNNTDecoder
from rnnt_joint import RNNTJoint
import copy

"""
Classes and methods from the nemo-toolkit for using the EncDecRNNTModel 
"""

class EncDecRNNTModel(LightningModule):
    def __init__(self, cfg: DictConfig):
        """
        Initialize encoder, decoder and joint from given config
        """
        super().__init__()
        self._cfg = cfg
        
        encoder_cfg = OmegaConf.to_container(self.cfg.encoder, resolve=True)
        del encoder_cfg["_target_"]
        self.encoder = ConformerEncoder(**encoder_cfg)
        
        decoder_cfg = OmegaConf.to_container(self.cfg.decoder, resolve=True)
        del decoder_cfg["_target_"]
        self.decoder = RNNTDecoder(**decoder_cfg)

        joint_cfg = OmegaConf.to_container(self.cfg.joint, resolve=True)
        del joint_cfg["_target_"]
        self.joint = RNNTJoint(**joint_cfg)

    def from_config_dict(config: 'DictConfig'):
        """Instantiates object using DictConfig-based configuration"""
        # Resolve the config dict
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
            config = OmegaConf.create(config)
            OmegaConf.set_struct(config, True)

        config = maybe_update_config_version(config)
        instance = EncDecRNNTModel(cfg=config)

        if not hasattr(instance, '_cfg'):
            instance._cfg = config
        return instance
    
    @property
    def cfg(self):
        return self._cfg

def _convert_config(cfg: 'OmegaConf'):
    # Get rid of cls -> _target_.
    if 'cls' in cfg and '_target_' not in cfg:
        cfg._target_ = cfg.pop('cls')

    # Get rid of params.
    if 'params' in cfg:
        params = cfg.pop('params')
        for param_key, param_val in params.items():
            cfg[param_key] = param_val

    # Recursion.
    try:
        for _, sub_cfg in cfg.items():
            if isinstance(sub_cfg, DictConfig):
                _convert_config(sub_cfg)
    except omegaconf_errors.OmegaConfBaseException as e:
        print(f"Skipped conversion for config/subconfig:\n{cfg}\n Reason: {e}.")

def maybe_update_config_version(cfg: 'DictConfig'):
    if cfg is not None and not isinstance(cfg, DictConfig):
        try:
            temp_cfg = OmegaConf.create(cfg)
            cfg = temp_cfg
        except omegaconf_errors.OmegaConfBaseException:
            # Cannot be cast to DictConfig, skip updating.
            return cfg

    # Make a copy of model config.
    cfg = copy.deepcopy(cfg)
    OmegaConf.set_struct(cfg, False)

    # Convert config.
    _convert_config(cfg)

    # Update model config.
    OmegaConf.set_struct(cfg, True)

    return cfg
    