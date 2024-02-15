import torch
from whisperspeech.pipeline import Pipeline

class WhisperSpeechModel(torch.nn.Module):
    def __init__(self):
        super(WhisperSpeechModel, self).__init__()
        self.pipe = Pipeline(s2a_ref='collabora/whisperspeech:s2a-q4-tiny-en+pl.model', torch_compile=True)

    def forward(self, text):
        with torch.no_grad():
            wav = self.pipe.generate(text)   
        return wav.cpu().numpy().flatten()
