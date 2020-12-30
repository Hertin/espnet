# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool
import logging
import math
import json
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.reshape(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            # print('InferenceBatchSoftmax input', input_.size())
            return F.softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("deepspeech model setting")

        group.add_argument(
            "--deepspeech2-rnn-hidden-size", default=768, type=int, help="Number of hidden dimension"
        )
        group.add_argument(
            "--deepspeech2-nb-layers", default=5, type=int, help=""
        )
        group.add_argument(
            "--deepspeech2-rnn-type", default="nn.LSTM", type=str, help=""
        )
        group.add_argument(
            "--deepspeech2-context", default=20, type=int, help=""
        )
        group.add_argument(
            "--deepspeech2-bidirectional", default=True, type=bool, help=""
        )
        group.add_argument(
            "--deepspeech2-signature-map", default="", type=str, help=""
        )
        group.add_argument(
            "--deepspeech2-init",
            type=str,
            default="pytorch",
            choices=[
                "pytorch",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
            ],
            help="how to initialize deepspeech parameters",
        )
        group.add_argument(
            "--adim",
            default=61,
            type=int,
            help="Number of intermediate output transformation dimensions",
        )
        group.add_argument(
            "--dropout-rate",
            default=0.0,
            type=float,
            help="Dropout rate for the encoder",
        )
        # group.add_argument(
        #     "--deepspeech2-beam-width", default=1, type=int, help=""
        # )
            # save_dir: data/librispeechclean/stft
    # sample_rate: 8000
    # input_dim: 161
    # window_size: 25
    # stride: 5
        
        return parser

    # @property
    # def attention_plot_class(self):
    #     """Return PlotAttentionReport."""
    #     return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        # torch.nn.Module.__init__(self)
        # if args.transformer_attn_dropout_rate is None:
        #     args.transformer_attn_dropout_rate = args.dropout_rate
        # self.encoder = Encoder(
        #     idim=idim,
        #     attention_dim=args.adim,
        #     attention_heads=args.aheads,
        #     linear_units=args.eunits,
        #     num_blocks=args.elayers,
        #     input_layer=args.transformer_input_layer,
        #     dropout_rate=args.dropout_rate,
        #     positional_dropout_rate=args.dropout_rate,
        #     attention_dropout_rate=args.transformer_attn_dropout_rate,
        # )
        # self.decoder = Decoder(
        #     odim=odim,
        #     attention_dim=args.adim,
        #     attention_heads=args.aheads,
        #     linear_units=args.dunits,
        #     num_blocks=args.dlayers,
        #     dropout_rate=args.dropout_rate,
        #     positional_dropout_rate=args.dropout_rate,
        #     self_attention_dropout_rate=args.transformer_attn_dropout_rate,
        #     src_attention_dropout_rate=args.transformer_attn_dropout_rate,
        # )
        # self.sos = odim - 1
        # self.eos = odim - 1
        # self.odim = odim
        # self.ignore_id = ignore_id
        # self.subsample = get_subsample(args, mode="asr", arch="transformer")
        # self.reporter = Reporter()

        # # self.lsm_weight = a
        # self.criterion = LabelSmoothingLoss(
        #     self.odim,
        #     self.ignore_id,
        #     args.lsm_weight,
        #     args.transformer_length_normalized_loss,
        # )
        # # self.verbose = args.verbose
        # self.reset_parameters(args)
        # self.adim = args.adim
        # self.mtlalpha = args.mtlalpha
        # if args.mtlalpha > 0.0:
        #     self.ctc = CTC(
        #         odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=False
        #     )
        # else:
        #     self.ctc = None

        # if args.report_cer or args.report_wer:
        #     self.error_calculator = ErrorCalculator(
        #         args.char_list,
        #         args.sym_space,
        #         args.sym_blank,
        #         args.report_cer,
        #         args.report_wer,
        #     )
        # else:
        #     self.error_calculator = None
        # self.rnnlm = None
        super(E2E, self).__init__()
        self.args = args
        # logging.warning(f'Pytorch version{torch.__version__}')
        # rnn_type=nn.LSTM, labels="abc", rnn_hidden_size=768, nb_layers=5, audio_conf=None,
        #          bidirectional=True, context=20

        # model metadata needed for serialization/deserialization
        # if audio_conf is None:
        #     audio_conf = {}
        self.hidden_size = self.args.deepspeech2_rnn_hidden_size # 768
        self.hidden_layers = self.args.deepspeech2_nb_layers # 5
        self.rnn_type = eval(self.args.deepspeech2_rnn_type) # nn.LSTM
        # self.audio_conf = self.config.feature
        self.context = self.args.deepspeech2_context # 20

        # with open(self.config.data.label_dir, 'r') as f:
        #     labels = json.load(f)
        self.labels = args.char_list
        self.bidirectional = self.args.deepspeech2_bidirectional

        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        # sample_rate = self.audio_conf.sample_rate # 8000
        # window_size = self.audio_conf.window_size / 1000.0 # 0.02 => 0.025
        self.adim = self.args.adim
        self.idim = idim
        self.num_classes = odim

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 1), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        # rnn_input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        rnn_input_size = self.idim
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        # print('rnn_input_size', rnn_input_size)
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=self.hidden_size, rnn_type=self.rnn_type,
                       bidirectional=self.bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self.hidden_layers - 1):
            rnn = BatchRNN(input_size=self.hidden_size, hidden_size=self.hidden_size, rnn_type=self.rnn_type,
                           bidirectional=self.bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(self.hidden_size, context=self.context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size),
            nn.Linear(self.hidden_size, self.adim, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()
        # self.PAD_token = self.config.data.PAD_token
        # if 'ctc' in self.config.model and self.config.model.ctc == 'warpctc':
        #     ## use warpctc
        #     print('deepspeech use warpctc')
        #     self.ctc_loss = CTCLoss()
        # else:
        #     ## use pytorch ctc
        #     print('deepspeech use pytorch-ctc')
        #     self.ctc_loss = nn.CTCLoss(blank=self.PAD_token, reduction='none')
        
        
        if args.report_cer or args.report_wer:
            self.error_calculator = ErrorCalculator(
                args.char_list,
                args.sym_space,
                args.sym_blank,
                args.report_cer,
                args.report_wer,
            )
        else:
            self.error_calculator = None
        # device = torch.device("cuda" if args.ngpu > 0 else "cpu")

        self.signature_map = torch.from_numpy(np.load(self.args.deepspeech2_signature_map).T).float()

        self.ctc = CTC(
            odim, args.adim, args.dropout_rate, 
            ctc_type=args.ctc_type, reduce=False, 
            ctc_lo=self.fc, signature_map=self.signature_map
        )

        self.reset_parameters(args)
        self.rnnlm = None
        self.reporter = Reporter()
        self.sos = odim - 1
        self.eos = odim - 1
        self.ignore_id = ignore_id
        # self.decoder = CTCBeamDecoder(
        #     labels=args.char_list, beam_width=args.beam_width
        # )

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.deepspeech2_init)

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) // m.stride[1] + 1)
        return seq_len.int()

    def forward(self, x, lengths, trns):
        '''
        :param torch.Tensor x: batch of padded source sequences (B, Tmax, idim)
        '''
        x = x.transpose(1,2).unsqueeze(1) # (B, 1, idim, Tmax)
        # logging.warning(f'DeepSpeech2 [x size] {x.size()}')
        # lengths = lengths.cpu().int()
        seq_len = self.get_seq_lens(lengths)
        # logging.warning(f'data type{ type(lengths) } {type(seq_len)} {lengths} {seq_len}')
        # print('output_lengths', output_lengths, x.size())
        x, _ = self.conv(x, seq_len.int())
        # logging.warning(f'DeepSpeech2 [CONV x size] {x.size()}')
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, seq_len.int())

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        # x = self.fc(x)
        # logging.warning(f'DeepSpeech2 [rnn x size] {x.size()}')

        # x = x @ self.signature_map.unsqueeze(0)

        # log_probs = x.log_softmax(dim=-1)
        x = x.transpose(0, 1)
        # target_lengths = trns.new([len(y[y != self.PAD_token]) for y in trns])
        # self.ctc(log_probs, hs_len, ys_pad)
        # logging.warning(f'Deepspeech [Size] { x.size()  } {seq_len.size()} {trns.size()} {trns}')
        loss_ctc_nonreduce = self.ctc(x, seq_len, trns,)
        loss_ctc_nonreduce[torch.isinf(loss_ctc_nonreduce)] = 0
        loss_ctc_nonreduce[torch.isnan(loss_ctc_nonreduce)] = 0
        loss_ctc = loss_ctc_nonreduce.mean()
        self.loss_ctc_nonreduce = loss_ctc_nonreduce
        # if self.error_calculator is not None:
        #     ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
        #     cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        # else:
        cer_ctc = None
        # loss = self.ctc_loss(log_probs, trns, output_lengths, target_lengths)
        # loss = loss.div(target_lengths.float())

        self.loss = loss_ctc
        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_data, loss_att=None, acc=None, cer_ctc=cer_ctc, cer=None, wer=None, mtl_loss=loss_data
            )
            # loss_att, acc, cer_ctc, cer, wer, mtl_loss
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        
        return self.loss_ctc_nonreduce

    # def scorers(self):
    #     """Scorers."""
    #     return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    # def encode(self, x):
    #     """Encode acoustic features.

    #     :param ndarray x: source acoustic feature (T, D)
    #     :return: encoder outputs
    #     :rtype: torch.Tensor
    #     """
    #     self.eval()
    #     x = torch.as_tensor(x).unsqueeze(0)
    #     enc_output, _ = self.encoder(x, None)
    #     return enc_output.squeeze(0)
    def encode(self, x, lengths):
        x = x.transpose(1,2).unsqueeze(1) # (B, 1, idim, Tmax)
        logging.warning(f'DeepSpeech2 [x size] {x.size()}')
        # lengths = lengths.cpu().int()
        seq_len = self.get_seq_lens(lengths)
        # logging.warning(f'data type{ type(lengths) } {type(seq_len)} {lengths} {seq_len}')
        # print('output_lengths', output_lengths, x.size())
        x, _ = self.conv(x, seq_len.int())
        # logging.warning(f'DeepSpeech2 [CONV x size] {x.size()}')
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, seq_len.int())

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        # x = self.fc(x)
        # logging.warning(f'DeepSpeech2 [rnn x size] {x.size()}')

        # x = x @ self.signature_map.unsqueeze(0)

        # log_probs = x.log_softmax(dim=-1)
        x = x.transpose(0, 1)
        # target_lengths = trns.new([len(y[y != self.PAD_token]) for y in trns])
        # self.ctc(log_probs, hs_len, ys_pad)
        # logging.warning(f'Deepspeech [Size] { x.size()  } {seq_len.size()} {trns.size()} {trns}')
        log_probs = self.ctc.log_softmax(x)

        
        return log_probs, seq_len
    # def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
    #     # """Recognize input speech.
        
    #     # :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
    #     # :param Namespace recog_args: argment Namespace contraining options
    #     # :param list char_list: list of characters
    #     # :param torch.nn.Module rnnlm: language model module
    #     # :return: N-best decoding results
    #     # :rtype: list
    #     # """

    #     enc_output = self.encode(x).unsqueeze(0)

    #     lpz = self.ctc.log_softmax(enc_output)
    #     lpz = lpz.squeeze(0)

    #     h = enc_output.squeeze(0)

    #     logging.info("input lengths: " + str(h.size(0)))
    #     # search parms
        
    #     penalty = recog_args.penalty

    #     # preprare sos
    #     y = self.sos
    #     vy = h.new_zeros(1).long()

    #     if recog_args.maxlenratio == 0:
    #         maxlen = h.shape[0]
    #     else:
    #         # maxlen >= 1
    #         maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
    #     minlen = int(recog_args.minlenratio * h.size(0))
    #     logging.info("max output length: " + str(maxlen))
    #     logging.info("min output length: " + str(minlen))

    #     # initialize hypothesis
    #     if rnnlm:
    #         hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
    #     else:
    #         hyp = {"score": 0.0, "yseq": [y]}
    #     if lpz is not None:
    #         ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
    #         hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
    #         hyp["ctc_score_prev"] = 0.0
    #         ctc_beam = lpz.shape[-1]

    #     hyps = [hyp]
    #     ended_hyps = []

    #     import six

    #     traced_decoder = None
    #     for i in six.moves.range(maxlen):
    #         logging.debug("position " + str(i))

    #         hyps_best_kept = []
    #         for hyp in hyps:
    #             vy[0] = hyp["yseq"][i]

    #             # get nbest local scores and their ids
    #             ys_mask = subsequent_mask(i + 1).unsqueeze(0)
    #             ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
    #             # FIXME: jit does not match non-jit result
    #             if use_jit:
    #                 if traced_decoder is None:
    #                     traced_decoder = torch.jit.trace(
    #                         self.decoder.forward_one_step, (ys, ys_mask, enc_output)
    #                     )
    #                 local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
    #             else:
    #                 local_att_scores = self.decoder.forward_one_step(
    #                     ys, ys_mask, enc_output
    #                 )[0]

    #             if rnnlm:
    #                 rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
    #                 local_scores = (
    #                     local_att_scores + recog_args.lm_weight * local_lm_scores
    #                 )
    #             else:
    #                 local_scores = local_att_scores

    #             if lpz is not None:
    #                 local_best_scores, local_best_ids = torch.topk(
    #                     local_att_scores, ctc_beam, dim=1
    #                 )
    #                 ctc_scores, ctc_states = ctc_prefix_score(
    #                     hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
    #                 )

    #                 local_scores = torch.from_numpy(
    #                     ctc_scores - hyp["ctc_score_prev"]
    #                 )
    #                 if rnnlm:
    #                     local_scores += (
    #                         recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
    #                     )
    #                 local_best_scores, joint_best_ids = torch.topk(
    #                     local_scores, beam, dim=1
    #                 )
    #                 local_best_ids = local_best_ids[:, joint_best_ids[0]]
    #             else:
    #                 local_best_scores, local_best_ids = torch.topk(
    #                     local_scores, beam, dim=1
    #                 )

    #             for j in six.moves.range(beam):
    #                 new_hyp = {}
    #                 new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
    #                 new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
    #                 new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
    #                 new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
    #                 if rnnlm:
    #                     new_hyp["rnnlm_prev"] = rnnlm_state
    #                 if lpz is not None:
    #                     new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
    #                     new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
    #                 # will be (2 x beam) hyps at most
    #                 hyps_best_kept.append(new_hyp)

    #             hyps_best_kept = sorted(
    #                 hyps_best_kept, key=lambda x: x["score"], reverse=True
    #             )[:beam]

    #         # sort and get nbest
    #         hyps = hyps_best_kept
    #         logging.debug("number of pruned hypothes: " + str(len(hyps)))
    #         if char_list is not None:
    #             logging.debug(
    #                 "best hypo: "
    #                 + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
    #             )

    #         # add eos in the final loop to avoid that there are no ended hyps
    #         if i == maxlen - 1:
    #             logging.info("adding <eos> in the last postion in the loop")
    #             for hyp in hyps:
    #                 hyp["yseq"].append(self.eos)

    #         # add ended hypothes to a final list, and removed them from current hypothes
    #         # (this will be a probmlem, number of hyps < beam)
    #         remained_hyps = []
    #         for hyp in hyps:
    #             if hyp["yseq"][-1] == self.eos:
    #                 # only store the sequence that has more than minlen outputs
    #                 # also add penalty
    #                 if len(hyp["yseq"]) > minlen:
    #                     hyp["score"] += (i + 1) * penalty
    #                     if rnnlm:  # Word LM needs to add final <eos> score
    #                         hyp["score"] += recog_args.lm_weight * rnnlm.final(
    #                             hyp["rnnlm_prev"]
    #                         )
    #                     ended_hyps.append(hyp)
    #             else:
    #                 remained_hyps.append(hyp)

    #         # end detection

    #         if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
    #             logging.info("end detected at %d", i)
    #             break

    #         hyps = remained_hyps
    #         if len(hyps) > 0:
    #             logging.debug("remeined hypothes: " + str(len(hyps)))
    #         else:
    #             logging.info("no hypothesis. Finish decoding.")
    #             break

    #         if char_list is not None:
    #             for hyp in hyps:
    #                 logging.debug(
    #                     "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
    #                 )

    #         logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

    #     nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
    #         : min(len(ended_hyps), recog_args.nbest)
    #     ]

    #     # check number of hypotheis
    #     if len(nbest_hyps) == 0:
    #         logging.warning(
    #             "there is no N-best results, perform recognition "
    #             "again with smaller minlenratio."
    #         )
    #         # should copy becasuse Namespace will be overwritten globally
    #         recog_args = Namespace(**vars(recog_args))
    #         recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
    #         return self.recognize(x, recog_args, char_list, rnnlm)

    #     logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
    #     logging.info(
    #         "normalized log probability: "
    #         + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
    #     )
    #     return nbest_hyps

    # def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
    #     """E2E attention calculation.

    #     :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
    #     :param torch.Tensor ilens: batch of lengths of input sequences (B)
    #     :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
    #     :return: attention weights with the following shape,
    #         1) multi-head case => attention weights (B, H, Lmax, Tmax),
    #         2) other case => attention weights (B, Lmax, Tmax).
    #     :rtype: float ndarray
    #     """
    #     with torch.no_grad():
    #         self.forward(xs_pad, ilens, ys_pad)
    #     ret = dict()
    #     for name, m in self.named_modules():
    #         if isinstance(m, MultiHeadedAttention):
    #             ret[name] = m.attn.cpu().numpy()
    #     return ret
