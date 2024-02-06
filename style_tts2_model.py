import sys
import os
subdirectory_path = os.path.join(os.path.dirname(__file__), 'StyleTTS2')
sys.path.append(subdirectory_path)

import wget
import yaml
import torch
import phonemizer
from utils import *
from models import *
from munch import Munch
from text_utils import TextCleaner
from contextlib import contextmanager
from nltk.tokenize import word_tokenize
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

"""
Wrapper to use the StyleTTS2 model and downloading the necessary weights and config files
"""

class StyleTTS2Model(torch.nn.Module):
    def __init__(self, styleTTS2_repo_path="StyleTTS2", device="cuda") -> None:
        """Initialize StyleTTS2 model for text sythesizing. StyleTTS2 repository have to be cloned locally and
           a valid path to the repo have to be passed
        Args:
            styleTTS2_repo_path: local or global path to the cloned StyleTTS2 repository
        Returns:
            Instnatce of the StyleTTS2 model 
        """ 
        super(StyleTTS2Model, self).__init__()

        self.device = device

        # setup text cleaner, phonemizer, and noise for inference
        self.textclenaer = TextCleaner()
        self.global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True,  with_stress=True)
        self.noise = torch.randn(1,1,256).to(self.device)
        
        # check if model weights exist and download them if not
        if not os.path.isdir(styleTTS2_repo_path+"/Models"):
            os.mkdir(styleTTS2_repo_path+"/Models")
        ljspeech_path = styleTTS2_repo_path+"/Models/LJSpeech/"
        if not os.path.isdir(ljspeech_path):
            os.mkdir(ljspeech_path)
        if not os.path.exists(ljspeech_path+"config.yml"):
            with change_dir(ljspeech_path):
                wget.download("https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/config.yml?download=true")
        if not os.path.exists(ljspeech_path+"epoch_2nd_00100.pth"):
             with change_dir(ljspeech_path):
                wget.download("https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth?download=true")
    
        with change_dir(styleTTS2_repo_path):
            config = yaml.safe_load(open("Models/LJSpeech/config.yml"))

            # load pretrained ASR model
            ASR_config = config.get('ASR_config', False)
            ASR_path = config.get('ASR_path', False)
            text_aligner = load_ASR_models(ASR_path, ASR_config)

            # load pretrained F0 model
            F0_path = config.get('F0_path', False)
            pitch_extractor = load_F0_models(F0_path)

            # load BERT model
            BERT_path = config.get('PLBERT_dir', False)
            plbert = load_plbert(BERT_path)

            # build model 
            self.model = build_model(recursive_munch(config['model_params']), text_aligner, pitch_extractor, plbert)
            _ = [self.model[key].eval() for key in self.model]
            _ = [self.model[key].to(self.device) for key in self.model]
            params_whole = torch.load("Models/LJSpeech/epoch_2nd_00100.pth", map_location='cpu')
            params = params_whole['net']
            for key in self.model:
                if key in params:
                    print('%s loaded' % key)
                    try:
                        self.model[key].load_state_dict(params[key])
                    except:
                        from collections import OrderedDict
                        state_dict = params[key]
                        new_state_dict = OrderedDict()
                        for k, v in state_dict.items():
                            name = k[7:] # remove `module.`
                            new_state_dict[name] = v
                        # load params
                        self.model[key].load_state_dict(new_state_dict, strict=False)
            #             except:
            #                 _load(params[key], model[key])
            _ = [self.model[key].eval() for key in self.model]
            
            # init sampler
            self.sampler = DiffusionSampler(
                self.model.diffusion.diffusion,
                sampler=ADPM2Sampler(),
                sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0), # empirical parameters
                clamp=False
            )

    def forward(self, text, diffusion_steps=5, embedding_scale=1):
        """Synthesize the entire given text. For good quality a whole sentence should be passed 
        Args:
            text: String of text that should be synthesized (should be whole sentence)
            embedding_scale:
            diffusion_steps:
        Returns:
            Synthesized audio data in wave format
        """ 
        
        text = text.strip()
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)

        tokens = self.textclenaer(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)

            t_en = self.model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self.model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = self.model.bert_encoder(bert_dur).transpose(-1, -2) 

            s_pred = self.sampler(self.noise, 
                embedding=bert_dur[0].unsqueeze(0), num_steps=diffusion_steps,
                embedding_scale=embedding_scale).squeeze(0)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            d = self.model.predictor.text_encoder(d_en, s, input_lengths, text_mask)

            x, _ = self.model.predictor.lstm(d)
            duration = self.model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_dur[-1] += 5

            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)

            # encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            F0_pred, N_pred = self.model.predictor.F0Ntrain(en, s)
            out = self.model.decoder((t_en @ pred_aln_trg.unsqueeze(0).to(self.device)), 
                                    F0_pred, N_pred, ref.squeeze().unsqueeze(0))
            
        return out.squeeze().cpu().numpy()
    
def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

@contextmanager
def change_dir(destination):
    """Temporarily change the working directory to the given path and change it back at the end 
        Args:
            destination: global or local path to a directory
    """
    try:
        cwd = os.getcwd()  # Save the current working directory
        os.chdir(destination)  # Change to the target directory
        yield
    finally:
        os.chdir(cwd)  # Change back to the original directory