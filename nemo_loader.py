
import os
import wget
import torch
import hashlib
import pathlib
import tarfile
import tempfile
from pathlib import Path 
from omegaconf import DictConfig, OmegaConf
from models_voice_assistant.STT.enc_dec_rnnt_model import EncDecRNNTModel

"""
Utility methods from the nemo-toolkit to download and use checkpoints from the NGC cloud
"""

def resolve_cache_dir():
    override_dir = os.environ.get("NEMO_CACHE_DIR", "")
    if override_dir == "":
        path = pathlib.Path.joinpath(pathlib.Path.home(), f'.cache/torch/NeMo/NeMo_1.21.0rc0')
    else:
        path = pathlib.Path(override_dir).resolve()
    return path

def maybe_download_from_cloud(url, filename, subfolder=None, cache_dir=None, refresh_cache=False):
    if cache_dir is None:
        cache_location = Path.joinpath(Path.home(), ".cache/torch/NeMo")
    else:
        cache_location = cache_dir
    if subfolder is not None:
        destination = Path.joinpath(cache_location, subfolder)
    else:
        destination = cache_location

    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    destination_file = Path.joinpath(destination, filename)

    if os.path.exists(destination_file):
        if refresh_cache:
            os.remove(destination_file)
        else:
            return str(destination_file)
    # download file
    wget_uri = url + filename

    # NGC links do not work everytime so we try and wait
    i = 0
    max_attempts = 3
    while i < max_attempts:
        i += 1
        try:
            wget.download(wget_uri, str(destination_file))
            if os.path.exists(destination_file):
                return destination_file
            else:
                return ""
        except:
            print(f"Download from cloud failed. Attempt {i} of {max_attempts}")
            continue
    raise ValueError("Not able to download url right now, please try again.")

def _get_ngc_pretrained_model_info(cloud_url, model_description):
    filename = cloud_url.split("/")[-1]
    url = cloud_url.replace(filename, "")
    cache_dir = pathlib.Path.joinpath(resolve_cache_dir(), f'{filename[:-5]}')
    # If either description and location in the cloud changes, this will force re-download
    cache_subfolder = hashlib.md5((cloud_url + model_description).encode('utf-8')).hexdigest()
    # if file exists on cache_folder/subfolder, it will be re-used, unless refresh_cache is True
    nemo_model_file_in_cache = maybe_download_from_cloud(
        url=url, filename=filename, cache_dir=cache_dir, subfolder=cache_subfolder, refresh_cache=False
    )
    return nemo_model_file_in_cache

def _unpack_nemo_file(path2file: str, out_folder: str, extract_config_only: bool = False) -> str:
    if not os.path.exists(path2file):
        raise FileNotFoundError(f"{path2file} does not exist")

    # we start with an assumption of uncompressed tar,
    # which should be true for versions 1.7.0 and above
    tar_header = "r:"
    try:
        tar_test = tarfile.open(path2file, tar_header)
        tar_test.close()
    except tarfile.ReadError:
        # can be older checkpoint => try compressed tar
        tar_header = "r:gz"
    tar = tarfile.open(path2file, tar_header)
    if not extract_config_only:
        tar.extractall(path=out_folder)
    else:
        members = [x for x in tar.getmembers() if ".yaml" in x.name]
        tar.extractall(path=out_folder, members=members)
    tar.close()
    return out_folder

def load_config_and_state_dict(
    _cls,
    restore_path: str,
    strict: bool = False,
):
    cwd = os.getcwd()
    if torch.cuda.is_available():
        map_location = torch.device('cuda')
    else:
        map_location = torch.device('cpu')

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Extract the nemo file into the temporary directory
            _unpack_nemo_file(
                path2file=restore_path, out_folder=tmpdir, extract_config_only=False
            )

            # Change current working directory to
            os.chdir(tmpdir)
            config_yaml = "model_config.yaml"

            if not isinstance(config_yaml, (OmegaConf, DictConfig)):
                conf = OmegaConf.load(config_yaml)
            else:
                conf = config_yaml
            
            # If override is top level config, extract just `model` from it
            if 'model' in conf:
                conf = conf.model
                return instance
            else:
                model_weights = os.path.join(tmpdir, "model_weights.ckpt")

            OmegaConf.set_struct(conf, True)
            os.chdir(cwd)

            instance = _cls.from_config_dict(config=conf)
            instance = instance.to(map_location)
            state_dict = torch.load(model_weights, map_location='cpu')
            instance.load_state_dict(state_dict, strict=strict)

        finally:
            os.chdir(cwd)

    return instance

def load_rnnt_model(cloud_url, model_description):
    nemo_model_file_in_cache = _get_ngc_pretrained_model_info(cloud_url=cloud_url, model_description=model_description)
    restore_path = os.path.abspath(os.path.expanduser(nemo_model_file_in_cache))
    model = load_config_and_state_dict(EncDecRNNTModel, restore_path)
    return model
