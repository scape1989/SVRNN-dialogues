import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sequential import MLP
sys.path.append("..")
from utils.sample import gumbel_softmax
from utils.loss import BPR_BOW_loss
from models.attention_module import Attn
import params


class VAECell(nn.Module):
    def __init__(self, state_is_tuple=True):
        super(VAECell, self).__init__()

        self._state_is_tuple = state_is_tuple
        # temperature of gumbel_softmax
        self.tau = nn.Parameter(torch.tensor([5.0]))

        self.enc_mlp = MLP(params.encoding_cell_size * 2 +
                           params.state_cell_size, [400, 200],
                           dropout_rate=params.dropout)
        self.enc_fc = nn.Linear(200, params.n_state)
        self.dec_mlp = MLP(params.n_state, [200, 200],
                           dropout_rate=params.dropout)

        self.dec_rnn_1 = nn.LSTM(params.embed_size,
                                 200 + params.n_state,
                                 1,
                                 batch_first=True)
        self.dec_fc_1 = nn.Linear(200 + params.n_state, params.max_vocab_cnt)
        self.dec_rnn_2 = nn.LSTM(params.embed_size,
                                 2 * (200 + params.n_state),
                                 1,
                                 batch_first=True)
        self.dec_fc_2 = nn.Linear(2 * (200 + params.n_state),
                                  params.max_vocab_cnt)

        self.bow_fc1 = nn.Linear(params.state_cell_size + 200, 400)
        self.bow_project1 = nn.Linear(400, params.max_vocab_cnt)
        self.bow_fc2 = nn.Linear(2 * (params.state_cell_size + 200), 400)
        self.bow_project2 = nn.Linear(400, params.max_vocab_cnt)
        if params.with_direct_transition:
            self.transit_mlp = MLP(params.n_state, [100, 100],
                                   dropout_rate=params.dropout)
        else:
            self.transit_mlp = MLP(params.state_cell_size, [100, 100],
                                   dropout_rate=params.dropout)
        self.transit_fc = nn.Linear(100, params.n_state)

        if params.cell_type == "gru":
            self.state_rnn = nn.GRUCell(params.encoding_cell_size * 2 + 200,
                                        params.state_cell_size)
        else:
            self.state_rnn = nn.LSTMCell(params.encoding_cell_size * 2 + 200,
                                         params.state_cell_size)

        # attention
        if params.use_sentence_attention:
            self.attn = Attn(params.attention_type,
                             params.encoding_cell_size * 2)

    def encode(self, inputs, h_prev, training=True):
        enc_inputs = torch.cat(
            [h_prev, inputs],
            1)  # [batch, encoding_cell_size * 2 + state_cell_size]
        net1 = self.enc_mlp(enc_inputs, training=training)
        logits_z = self.enc_fc(net1)
        q_z = F.softmax(logits_z, dim=1)
        log_q_z = F.log_softmax(logits_z, dim=1)

        return logits_z, q_z, log_q_z

    def context_encode(self, inputs, h_prev, prev_embeddings, training=True):
        '''
        :param inputs: sentence encoding for current dialogue index(utt) [batch, encoding_cell_size * 2]
        :param h_prev: previous h state from LSTM [batch, state_cell_size]
        :param prev_embeddings: previous sentence embeddings [batch, current_utt_index - 1, encoding_cell_size * 2]
        :return: hidden_state from vae [batch x n_state]
        '''
        attn_weights = self.attn(inputs, prev_embeddings)
        context = attn_weights.bmm(prev_embeddings).squeeze(1)
        enc_inputs = torch.cat(
            [h_prev, context],
            1)  # [batch, encoding_cell_size * 2 + state_cell_size]

        net1 = self.enc_mlp(enc_inputs, training=training)
        logits_z = self.enc_fc(net1)
        q_z = F.softmax(logits_z, dim=1)
        log_q_z = F.log_softmax(logits_z, dim=1)

        return logits_z, q_z, log_q_z

    def decode(self,
               z_samples,
               h_prev,
               dec_input_embedding,
               training=True,
               z_samples_context=None):
        net2 = self.dec_mlp(z_samples, training=training)  # [batch, 200]
        # decoder for user utterance
        dec_input_1 = torch.unsqueeze(
            torch.cat([h_prev, net2], dim=1),
            dim=0)  # [num_layer(1), batch, state_cell_size + 200]

        # decoder from context
        if params.use_sentence_attention:
            net2_context = self.dec_mlp(z_samples_context,
                                        training=training)  # [batch, 200]
            dec_input_1_context = torch.unsqueeze(
                torch.cat([h_prev, net2_context], dim=1),
                dim=0)  # [num_layer(1), batch, state_cell_size + 200]
        else:
            dec_input_1_context = dec_input_1

        dec_input_embedding[0] = dec_input_embedding[
            0][:, 0:-1, :]  # batch x (40 - 1) x 300
        dec_input_embedding[1] = dec_input_embedding[1][:, 0:-1, :]

        dec_outs_1, final_state_1 = self.dec_rnn_1(
            dec_input_embedding[0], (dec_input_1, dec_input_1_context))

        dec_outs_1 = F.dropout(dec_outs_1, p=params.dropout, training=training)
        dec_outs_1 = self.dec_fc_1(dec_outs_1)
        dec_input_2_h = torch.cat(
            [dec_input_1, final_state_1[0]],
            dim=2)  # [1, batch, 2 * (state_cell_size + 200)]

        dec_input_2_c = torch.cat(
            [dec_input_1, final_state_1[1]],
            dim=2)  # [1, batch, 2 * (state_cell_size + 200)]

        dec_outs_2, final_state_2 = self.dec_rnn_2(
            dec_input_embedding[1], (dec_input_2_h, dec_input_2_c))
        dec_outs_2 = F.dropout(dec_outs_2, p=params.dropout, training=training)
        dec_outs_2 = self.dec_fc_2(dec_outs_2)

        # for computing BOW loss
        bow_logits1 = bow_logits2 = None
        if params.with_BOW:
            bow_fc1 = self.bow_fc1(torch.squeeze(dec_input_1, dim=0))
            bow_fc1 = torch.tanh(bow_fc1)
            if params.dropout > 0:
                bow_fc1 = F.dropout(bow_fc1,
                                    p=params.dropout,
                                    training=training)
            bow_logits1 = self.bow_project1(
                bow_fc1)  # [batch_size, vocab_size]

            bow_fc2 = self.bow_fc2(torch.squeeze(dec_input_2_h, dim=0))
            bow_fc2 = torch.tanh(bow_fc2)
            if params.dropout > 0:
                bow_fc2 = F.dropout(bow_fc2,
                                    p=params.dropout,
                                    training=training)
            bow_logits2 = self.bow_project2(bow_fc2)
        return net2, dec_outs_1, dec_outs_2, bow_logits1, bow_logits2

    def forward(self,
                inputs,
                state,
                dec_input_embedding,
                dec_seq_lens,
                output_tokens,
                prev_z_t=None,
                training=True,
                prev_embeddings=None):
        if params.with_direct_transition:
            assert prev_z_t is not None
        if self._state_is_tuple:
            (h_prev, _) = state
        else:
            h_prev = state
        # encode
        logits_z, q_z, log_q_z = self.encode(inputs, h_prev, training=training)

        # sample
        z_samples, logits_z_samples = gumbel_softmax(
            logits_z, self.tau, hard=False)  # [batch, n_state]

        #encode from context
        if params.use_sentence_attention:
            logits_z_context, q_z_context, log_q_z_context = self.context_encode(
                inputs, h_prev, prev_embeddings, training=training)

            #sample from context
            z_samples_context, logits_z_samples_context = gumbel_softmax(
                logits_z_context, self.tau, hard=False)  # [batch, n_state]
        else:
            z_samples_context = None

        # decode
        net2, dec_outs_1, dec_outs_2, bow_logits1, bow_logits2 = self.decode(
            z_samples,
            h_prev,
            dec_input_embedding,
            training=training,
            z_samples_context=z_samples_context)

        if params.with_direct_transition:
            net3 = self.transit_mlp(prev_z_t, training=training)
            p_z = self.transit_fc(net3)
            p_z = F.softmax(p_z, dim=1)
            log_p_z = torch.log(p_z + 1e-20)

        else:
            net3 = self.transit_mlp(h_prev, training=training)
            p_z = self.transit_fc(net3)
            p_z = F.softmax(p_z, dim=1)
            log_p_z = torch.log(p_z + 1e-20)

        recur_input = torch.cat([net2, inputs],
                                dim=1)  # [batch, encoding_cell_size * 2 + 200]
        next_state = self.state_rnn(recur_input, state)

        losses = BPR_BOW_loss(output_tokens,
                              dec_outs_1,
                              dec_outs_2,
                              log_p_z,
                              log_q_z,
                              p_z,
                              q_z,
                              bow_logits1=bow_logits1,
                              bow_logits2=bow_logits2)

        return losses, z_samples, next_state, p_z, bow_logits1, bow_logits2
