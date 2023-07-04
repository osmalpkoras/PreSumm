import torch
from torch import nn

import torch.nn.functional as F


class Bass(nn.Module):
    def __init__(self, model):
        super(Bass, self).__init__()
        self.model = model
        self.first_call = True
        self.src_features = None

    def forward(self, encoder_input_ids, encoder_padding, segmentations, decoder_input_ids, decoder_padding, step, dec_states=None):
        if self.first_call:
            self.src_features = self.model.bert(encoder_input_ids, segmentations, encoder_padding)
            dec_states = self.model.decoder.init_decoder_state(encoder_input_ids, self.src_features, with_cache=True)
            self.first_call = False
            
        dec_out, dec_states = self.model.decoder(decoder_input_ids, self.src_features, dec_states, step=step)
        output = self.model.generator[0](dec_out)
        return None, output[:,-1,:], dec_states

