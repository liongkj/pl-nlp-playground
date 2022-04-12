import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import BLEUScore

from components import AttnDecoderRNN, EncoderRnn

SOS_token = 0
EOS_token = 1
teacher_forcing_ratio = 0.5
MAX_LENGTH = 10
class BasicModel(pl.LightningModule):
    def __init__(self, input_lang, output_lang,hidden_size = 256,max_length = MAX_LENGTH, learning_rate = 0.001):
        super(BasicModel, self).__init__()
        self.encoder = EncoderRnn(input_lang.n_words, hidden_size)
        
        self.decoder = AttnDecoderRNN(hidden_size, output_lang.n_words,dropout_p=0.1,max_length=max_length)
        
        self.decoder.initHidden()
        self.save_hyperparameters()
        self.criterion = nn.NLLLoss()
        self.metric = BLEUScore()

    def forward(self, x, label):
        
        x_length, label_length = x.size(0), label.size(0)
        out_list = torch.zeros(self.hparams.max_length,self.encoder.hidden_size, device=self.device)
        # print(out_list.shape)
        encoder_hidden = self.encoder.initHidden()
        for idx in range(x_length):
            encoder_out, encoder_hidden = self.encoder(x[idx], encoder_hidden)
            out_list[idx] = encoder_out[0,0]

        
        decoder_input = torch.tensor([[SOS_token]],device=self.device)  # SOS
        decoder_hidden = encoder_hidden
        return decoder_input, decoder_hidden, out_list

    def training_step(self, batch, batch_idx):
        x,label = batch
        x = x.squeeze(0)
        label = label.squeeze(0)
        x_length, label_length = x.size(0), label.size(0)
        loss = 0
        decoder_input, decoder_hidden, out_list = self(x,label)
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
            for di in range(label_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, out_list)
                loss += self.criterion(decoder_output, label[di])
                decoder_input = label[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(label_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, out_list)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, label[di])
                if decoder_input.item() == EOS_token:
                    break
        return loss

    
    def validation_step(self, batch, batch_idx):
        x,label = batch
        x = x.squeeze()
        label = label.squeeze()
        lab = []
        for l in label:
            lab.append(self.hparams.output_lang.index2word[l.item()])
        decoder_input, decoder_hidden, out_list = self(x,label)
        decoded_words = []
        decoder_attentions = torch.zeros(self.hparams.max_length, self.hparams.max_length)

        for di in range(self.hparams.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, out_list)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(self.hparams.output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
        print(self.metric(decoded_words,lab))
        print(f"{decoded_words}>>{lab}")

        return decoded_words, decoder_attentions[:di + 1]
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        
        return optimizer

    # def validation_step(self, batch, batch_idx):
    #     input_tensor, target_tensor = batch
    #     encoder_outputs, encoder_hidden = self.encoder(input_tensor)
    #     decoder_hidden = encoder_hidden[:self.decoder.n_layers]
    #     decoder_input = torch.tensor([[SOS_token]])  # SOS
    #     decoder_output, decoder_hidden, attn_weights = self.decoder(
    #         decoder_input, decoder_hidden, encoder_outputs)
