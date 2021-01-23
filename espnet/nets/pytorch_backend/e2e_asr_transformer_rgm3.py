# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
import logging
import math

import numpy
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
import chainer
from chainer import reporter
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.argument import (
    add_arguments_transformer_common,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.encoder import Encoder, Extractor, EncoderLang
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
import pickle


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""
    def report(self, loss_f, loss_h, cer_ctc_h, cer_ctc_f, loss_f_fake, cer_ctc_fake, cer, wer, mtl_loss, penalty):
        """Report at every step."""
        reporter.report({"loss_f": loss_f}, self)
        reporter.report({"loss_h": loss_h}, self)
        reporter.report({"cer_ctc_h": cer_ctc_h}, self)
        reporter.report({"cer_ctc_f": cer_ctc_f}, self)
        reporter.report({"loss_f_fake": loss_f_fake}, self)
        reporter.report({"cer_ctc_fake": cer_ctc_fake}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)
        reporter.report({"penalty": penalty}, self)

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group = add_arguments_transformer_common(group)

        return parser

    @property
    def attention_plot_class(self):
        """Return PlotAttentionReport."""
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        # fill missing arguments for compatibility
        args = fill_missing_args(args, self.add_arguments)
        self.args = args

        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.encoder_f = EncoderLang(
            idim=args.adim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers - args.elayers_extractor,
            input_layer='Identity',
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            num_langs=args.num_langs
        )
        self.encoder_h = Encoder(
            idim=args.adim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers -args.elayers_extractor,
            input_layer='Identity',
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
        )
        if args.extractor_lf == False:
            if args.elayers_extractor == 0:
                self.encoder_phi = Extractor(
                    idim=idim,
                    attention_dim=args.adim,
                    attention_heads=args.aheads,
                    linear_units=args.eunits,
                    num_blocks=args.elayers,
                    input_layer='conv2d',
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    attention_dropout_rate=args.transformer_attn_dropout_rate,
                )
            else:
                self.encoder_phi = Encoder(
                    idim=idim,
                    attention_dim=args.adim,
                    attention_heads=args.aheads,
                    linear_units=args.eunits,
                    num_blocks=args.elayers_extractor,
                    input_layer='conv2d',
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    attention_dropout_rate=args.transformer_attn_dropout_rate,
                )
        else:
            if args.elayers_extractor == 0:
                self.encoder_phi = ExtractorLang(
                    idim=idim,
                    attention_dim=args.adim,
                    attention_heads=args.aheads,
                    linear_units=args.eunits,
                    num_blocks=args.elayers,
                    input_layer='conv2d',
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    attention_dropout_rate=args.transformer_attn_dropout_rate,
                    num_langs=args.num_langs_f
                )
            else:
                self.encoder_phi = EncoderLang(
                    idim=idim,
                    attention_dim=args.adim,
                    attention_heads=args.aheads,
                    linear_units=args.eunits,
                    num_blocks=args.elayers_extractor,
                    input_layer='conv2d',
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    attention_dropout_rate=args.transformer_attn_dropout_rate,
                    num_langs=args.num_langs_f,
                    condition_ex=True
                )
        self.decoder = None
        self.criterion = None
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = get_subsample(args, mode="asr", arch="transformer")
        self.reporter = Reporter()

        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        self.phs_aware = args.phs_aware
        if self.phs_aware:
            with open(args.phs_aware_dict, 'rb') as f:
                self.phs_aware_dict = pickle.load(f)
        else:
            self.phs_aware_dict = None
        self.ctc_f = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=False,langid2phs = self.phs_aware_dict
        )
        self.ctc_h = CTC(
            odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=False,langid2phs = self.phs_aware_dict
        )
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
        self.rnnlm = None
        self.clamp = torch.nn.ReLU()
        self.rgm_lambda = args.rgm_lambda
        self.num_langs = args.num_langs
        self.sampling_lf = args.sampling_lf
        self.extractor_lf = args.extractor_lf
        if self.extractor_lf:
            with open(args.lf_dict_dir_model, 'rb') as f:
                self.lang_family = pickle.load(f)


    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, langs, xs_pad, ilens, ys_pad, langs_f, step, cc= False):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # print(step)
        if step == '1f':
            self.encoder_phi.requires_grad_(False)
            self.encoder_h.requires_grad_(False)
            self.ctc_h.requires_grad_(False)
            self.encoder_f.requires_grad_(True)
            self.ctc_f.requires_grad_(True)
        elif step == '2h':
            self.encoder_phi.requires_grad_(False)
            self.encoder_h.requires_grad_(True)
            self.ctc_h.requires_grad_(True)
            self.encoder_f.requires_grad_(False)
            self.ctc_f.requires_grad_(False)
        else:
            self.encoder_phi.requires_grad_(True)
            self.encoder_h.requires_grad_(False)
            self.ctc_h.requires_grad_(False)
            self.encoder_f.requires_grad_(False)
            self.ctc_f.requires_grad_(False)
        # 1. forward encoder
        batch_size = xs_pad.size(0)
        ys = [y[y != self.ignore_id] for y in ys_pad]
        olens = torch.from_numpy(numpy.fromiter((x.size(0) for x in ys), dtype=numpy.int32))

        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel



        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        # extract from feature extractor phi
        if not self.extractor_lf:
            es_pad, es_mask = self.encoder_phi(xs_pad, src_mask)
        else:
            es_pad, es_mask = self.encoder_phi(langs_f, xs_pad, src_mask)

        if step == '1f':
            hs_pad_f, hs_mask_f = self.encoder_f(langs, es_pad, es_mask)
            self.hs_pad = hs_pad_f

            hs_len = hs_mask_f.view(batch_size, -1).sum(1)
            valid_indices = hs_len.cpu().int() > olens
            invalid = False
            if torch.sum(valid_indices) < batch_size:
                invalid = True

            if cc == True:
                hs_pad_h_check, hs_mask_h_check = self.encoder_h(es_pad, es_mask)
        elif step == '2h':
            hs_pad_h, hs_mask_h = self.encoder_h(es_pad, es_mask)
            self.hs_pad = hs_pad_h

            hs_len = hs_mask_h.view(batch_size, -1).sum(1)
            valid_indices = hs_len.cpu().int() > olens
            invalid = False
            if torch.sum(valid_indices) < batch_size:
                invalid = True

            if cc == True:
                hs_pad_f_check, hs_mask_f_check = self.encoder_f(langs, es_pad, es_mask)
        else:
            num_langs = langs.size(1)
            fake_langs = langs.clone()
            for batch_idx in range(len(fake_langs)):
                lang_idx = torch.argmax(fake_langs[batch_idx])
                fake_langs[batch_idx][lang_idx] = 0
                if not self.sampling_lf:
                    fake_lang_idx = numpy.random.choice([a for a in range(num_langs) if a != lang_idx])
                else:
                    fake_lang_idx = numpy.random.choice(self.lang_family[int(lang_idx)])
                fake_langs[batch_idx][fake_lang_idx] = 1
            #print('hs pad f real fake:')
            hs_pad_f_real, hs_mask_f_real = self.encoder_f(langs, es_pad, es_mask)
            hs_pad_f_fake, hs_mask_f_fake = self.encoder_f(fake_langs, es_pad, es_mask)
            hs_pad_h, hs_mask_h = self.encoder_h(es_pad, es_mask)
            self.hs_pad = hs_pad_h

            # remove utterances that are shorter than target
            hs_len = hs_mask_h.view(batch_size, -1).sum(1)
            valid_indices = hs_len.cpu().int() > olens
            invalid = False
            if torch.sum(valid_indices) < batch_size:
                invalid = True


        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats

        if step == '1f':
            # filter invalid CTC loss for classifier f
            lang_ids = None
            if self.phs_aware:
                lang_ids = torch.argmax(langs, dim=-1, keepdim = True)

            loss_ctc_nonreduce_f = self.ctc_f(hs_pad_f.view(batch_size, -1, self.adim), hs_len, ys_pad, lang_ids = lang_ids)
            invalid_idx_f = torch.isinf(loss_ctc_nonreduce_f) | torch.isnan(loss_ctc_nonreduce_f) | (loss_ctc_nonreduce_f < 0) | (loss_ctc_nonreduce_f > CTC_LOSS_THRESHOLD)
            # print('Step 1 Loss')
            # print(xs_pad.size())
            # print(invalid_idx_f, invalid)
            # if torch.sum(invalid_idx_f != 0):
            #     print(f'Step 1: Invalid ctc loss for classifier f {invalid} num invalid {torch.sum(invalid_idx_f != 0)} {loss_ctc_nonreduce_f[invalid_idx_f]}')

            loss_ctc_nonreduce_f[invalid_idx_f] = 0
            # loss_ctc_nonreduce[torch.isnan(loss_ctc_nonreduce)] = 0
            loss_ctc = loss_ctc_nonreduce_f[loss_ctc_nonreduce_f!=0].mean() if any(loss_ctc_nonreduce_f!=0) else 0

            # print('Step 1 Loss Check (f): ')
            # print(loss_ctc_nonreduce_f)

            if cc == True:
                print('Step 1 Loss Check (h): ')
                loss_ctc_nonreduce_h_check = self.ctc_h(hs_pad_h_check.view(batch_size, -1, self.adim), hs_len, ys_pad,lang_ids = lang_ids)
                invalid_idx_h_check = torch.isinf(loss_ctc_nonreduce_h_check) | torch.isnan(loss_ctc_nonreduce_h_check) | (loss_ctc_nonreduce_h_check < 0) | (loss_ctc_nonreduce_h_check > CTC_LOSS_THRESHOLD)
                print(loss_ctc_nonreduce_h_check)

        elif step == '2h':
            lang_ids = None
            if self.phs_aware:
                lang_ids = torch.argmax(langs, dim=-1, keepdim = True)
            # filter invalid CTC loss for classifier h
            loss_ctc_nonreduce_h = self.ctc_h(hs_pad_h.view(batch_size, -1, self.adim), hs_len, ys_pad,lang_ids = lang_ids)
            invalid_idx_h = torch.isinf(loss_ctc_nonreduce_h) | torch.isnan(loss_ctc_nonreduce_h) | (loss_ctc_nonreduce_h < 0) | (loss_ctc_nonreduce_h > CTC_LOSS_THRESHOLD)
            # print('Step 2 Loss')
            # print(xs_pad.size())
            # print(invalid_idx_h, invalid)
            # if torch.sum(invalid_idx_h != 0):
            #     print(f'Step 2: Invalid ctc loss for classifier h {invalid} num invalid {torch.sum(invalid_idx_h != 0)} {loss_ctc_nonreduce_h[invalid_idx_h]}')
            loss_ctc_nonreduce_h[invalid_idx_h] = 0
            # loss_ctc_nonreduce[torch.isnan(loss_ctc_nonreduce)] = 0
            loss_ctc = loss_ctc_nonreduce_h[loss_ctc_nonreduce_h!=0].mean() if any(loss_ctc_nonreduce_h!=0) else 0

            # print('Step 2 Loss Check (h): ')
            # print(loss_ctc_nonreduce_h)

            if cc == True:
                lang_ids = None
                if self.phs_aware:
                    lang_ids = torch.argmax(langs, dim=-1, keepdim = True)
                print('Step 2 Loss Check (f real): ')
                loss_ctc_nonreduce_f_check = self.ctc_f(hs_pad_f_check.view(batch_size, -1, self.adim), hs_len, ys_pad, lang_ids = lang_ids)
                invalid_idx_f_check = torch.isinf(loss_ctc_nonreduce_f_check) | torch.isnan(loss_ctc_nonreduce_f_check) | (loss_ctc_nonreduce_f_check < 0) | (loss_ctc_nonreduce_f_check > CTC_LOSS_THRESHOLD)
                print(loss_ctc_nonreduce_f_check)
        else:
            lang_ids = None
            if self.phs_aware:
                real_lang_ids = torch.argmax(langs, dim=-1, keepdim = True)
                fake_lang_ids = torch.argmax(fake_langs, dim=-1, keepdim = True)
                lang_ids = torch.cat([real_lang_ids, fake_lang_ids], dim = -1)
            loss_ctc_nonreduce_fake = self.ctc_f(hs_pad_f_fake.view(batch_size, -1, self.adim), hs_len, ys_pad, lang_ids = lang_ids)
            invalid_idx_fake = torch.isinf(loss_ctc_nonreduce_fake) | torch.isnan(loss_ctc_nonreduce_fake) | (loss_ctc_nonreduce_fake < 0) | (loss_ctc_nonreduce_fake > CTC_LOSS_THRESHOLD)
            # if torch.sum(invalid_idx_fake != 0):
            #     print(f'Step 3: Invalid fake ctc loss for classifier f {invalid} num invalid {torch.sum(invalid_idx_fake != 0)} {loss_ctc_nonreduce_fake[invalid_idx_fake]}')

            loss_ctc_nonreduce_real = self.ctc_f(hs_pad_f_real.view(batch_size, -1, self.adim), hs_len, ys_pad, lang_ids = lang_ids)
            invalid_idx_real = torch.isinf(loss_ctc_nonreduce_real) | torch.isnan(loss_ctc_nonreduce_real) | (loss_ctc_nonreduce_real < 0) | (loss_ctc_nonreduce_real > CTC_LOSS_THRESHOLD)
            # if torch.sum(invalid_idx_real != 0):
            #     print(f'Step 3: Invalid real ctc loss for classifier f {invalid} num invalid {torch.sum(invalid_idx_real != 0)} {loss_ctc_nonreduce_real[invalid_idx_real]}')

            loss_ctc_nonreduce_h = self.ctc_h(hs_pad_h.view(batch_size, -1, self.adim), hs_len, ys_pad, lang_ids = real_lang_ids)
            invalid_idx_h = torch.isinf(loss_ctc_nonreduce_h) | torch.isnan(loss_ctc_nonreduce_h) | (loss_ctc_nonreduce_h < 0) | (loss_ctc_nonreduce_h > CTC_LOSS_THRESHOLD)
            # if torch.sum(invalid_idx_h != 0):
            #     print(f'Step 3: Invalid ctc loss for classifier h {invalid} num invalid {torch.sum(invalid_idx_h != 0)} {loss_ctc_nonreduce_h[invalid_idx_h]}')

            # print('Step 3 Loss check (f_fake, f_real, h) ')
            # print(loss_ctc_nonreduce_fake, loss_ctc_nonreduce_real, loss_ctc_nonreduce_h)

            loss_ctc_nonreduce_fake[invalid_idx_fake] = 0
            loss_ctc_nonreduce_fake[invalid_idx_real] = 0
            loss_ctc_nonreduce_fake[invalid_idx_h] = 0

            loss_ctc_nonreduce_real[invalid_idx_fake] = 0
            loss_ctc_nonreduce_real[invalid_idx_real] = 0
            loss_ctc_nonreduce_real[invalid_idx_h] = 0

            loss_ctc_nonreduce_h[invalid_idx_fake] = 0
            loss_ctc_nonreduce_h[invalid_idx_real] = 0
            loss_ctc_nonreduce_h[invalid_idx_h] = 0

            # loss_ctc_nonreduce[torch.isnan(loss_ctc_nonreduce)] = 0
            loss_ctc_fake = loss_ctc_nonreduce_fake[loss_ctc_nonreduce_fake!=0].mean() if any(loss_ctc_nonreduce_fake!=0) else 0
            loss_ctc_real = loss_ctc_nonreduce_real[loss_ctc_nonreduce_real!=0].mean() if any(loss_ctc_nonreduce_real!=0) else 0
            loss_ctc = loss_ctc_nonreduce_h[loss_ctc_nonreduce_h!=0].mean() if any(loss_ctc_nonreduce_h!=0) else 0

        cer_ctc_f = None
        cer_ctc_h = None
        cer_ctc_fake = None
        if not self.training and self.error_calculator is not None:
            lang_ids = None
            if self.phs_aware:
                real_lang_ids = torch.argmax(langs, dim=-1, keepdim = True)
                fake_lang_ids = torch.argmax(fake_langs, dim=-1, keepdim = True)
                lang_ids = torch.cat([real_lang_ids, fake_lang_ids], dim = -1)

            ys_hat_h = self.ctc_h.argmax(hs_pad_h.view(batch_size, -1, self.adim),lang_ids = real_lang_ids).data
            cer_ctc_h = self.error_calculator(ys_hat_h[loss_ctc_nonreduce_h!=0].cpu(), ys_pad.cpu(), is_ctc=True)
            ys_hat_f = self.ctc_f.argmax(hs_pad_f_real.view(batch_size, -1, self.adim), lang_ids = lang_ids).data
            cer_ctc_f = self.error_calculator(ys_hat_f[loss_ctc_nonreduce_real!=0].cpu(), ys_pad.cpu(), is_ctc=True)
            ys_hat_fake = self.ctc_f.argmax(hs_pad_f_fake.view(batch_size, -1, self.adim), lang_ids = lang_ids).data
            cer_ctc_fake = self.error_calculator(ys_hat_fake[loss_ctc_nonreduce_fake!=0].cpu(), ys_pad.cpu(), is_ctc=True)
        # for visualization
        if not self.training:
            lang_ids = None
            if self.phs_aware:
                lang_ids = torch.argmax(langs, dim=-1, keepdim = True)
            self.ctc_h.softmax(hs_pad_h,lang_ids = lang_ids)
            self.ctc_f.softmax(hs_pad_f_real, lang_ids = lang_ids)

        penalty = None
        if step == '3p':
            penalty = loss_ctc_fake - loss_ctc_real

        if step == '1f' or step == '2h':
            self.loss = loss_ctc
            # print('Step 1 or 2 loss: ', self.loss)
        else:
            self.loss = loss_ctc + self.rgm_lambda * self.clamp(penalty)
            # print('Step 3 loss: ', self.loss)

        loss_ctc_f_data = None
        loss_ctc_h_data = None
        loss_ctc_fake_data = None
        if step == '1f':
            loss_ctc_f_data = float(loss_ctc)
        if step == '2h':
            loss_ctc_h_data = float(loss_ctc)
        if step == '3p':
            loss_ctc_f_data = float(loss_ctc_real)
            loss_ctc_h_data = float(loss_ctc)
            loss_ctc_fake_data = float(loss_ctc_fake)

        loss_data = float(self.loss)

        penalty_data = None
        if step == '3p':
            penalty_data = float(penalty)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            if step == '3p':
                self.reporter.report(
                    loss_ctc_f_data, loss_ctc_h_data, cer_ctc_h, cer_ctc_f, loss_ctc_fake_data, cer_ctc_fake, None, None, loss_data, penalty_data
                )
        else:
            print("loss (=%f) is not correct", loss_data)
            print('Step: ', step)
            if step == '1f':
                print(loss_ctc_f_data)
            if step == '2h':
                print(loss_ctc_h_data)
            if step == '3p':
                print(loss_ctc_f_data, loss_ctc_h_data, loss_ctc_fake_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc_h, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        x = self.encoder_phi(x, None)
        enc_output, _ = self.encoder_h(x, None)
        return enc_output.squeeze(0)

    def encode_with_length(self, xs_pad, ilens):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        es_pad, es_mask = self.encoder_phi(xs_pad, src_mask)
        hs_pad, hs_mask = self.encoder_h(es_pad, es_mask)
        self.hs_pad = hs_pad
        # x = torch.as_tensor(x).unsqueeze(0)
        # enc_output, hs_mask = self.encoder(x, None)
        batch_size = xs_pad.size(0)
        hs_len = hs_mask.view(batch_size, -1).sum(1)
        log_probs = self.ctc_h.log_softmax(hs_pad.view(batch_size, -1, self.adim))
        return log_probs, hs_len

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """
        enc_output = self.encode(x).unsqueeze(0)
        if self.mtlalpha == 1.0:
            recog_args.ctc_weight = 1.0
            logging.info("Set to pure CTC decoding mode.")

        if self.mtlalpha > 0 and recog_args.ctc_weight == 1.0:
            from itertools import groupby

            lpz = self.ctc_h.argmax(enc_output)
            collapsed_indices = [x[0] for x in groupby(lpz[0])]
            hyp = [x for x in filter(lambda x: x != self.blank, collapsed_indices)]
            nbest_hyps = [{"score": 0.0, "yseq": [self.sos] + hyp}]
            if recog_args.beam_size > 1:
                raise NotImplementedError("Pure CTC beam search is not implemented.")
            # TODO(hirofumi0810): Implement beam search
            return nbest_hyps
        elif self.mtlalpha > 0 and recog_args.ctc_weight > 0.0:
            lpz = self.ctc_h.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info("input lengths: " + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info("max output length: " + str(maxlen))
        logging.info("min output length: " + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {"score": 0.0, "yseq": [y], "rnnlm_prev": None}
        else:
            hyp = {"score": 0.0, "yseq": [y]}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp["ctc_state_prev"] = ctc_prefix_score.initial_state()
            hyp["ctc_score_prev"] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six

        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug("position " + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp["yseq"][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp["yseq"]).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(
                            self.decoder.forward_one_step, (ys, ys_mask, enc_output)
                        )
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(
                        ys, ys_mask, enc_output
                    )[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp["rnnlm_prev"], vy)
                    local_scores = (
                        local_att_scores + recog_args.lm_weight * local_lm_scores
                    )
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1
                    )
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp["yseq"], local_best_ids[0], hyp["ctc_state_prev"]
                    )
                    local_scores = (1.0 - ctc_weight) * local_att_scores[
                        :, local_best_ids[0]
                    ] + ctc_weight * torch.from_numpy(
                        ctc_scores - hyp["ctc_score_prev"]
                    )
                    if rnnlm:
                        local_scores += (
                            recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                        )
                    local_best_scores, joint_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(
                        local_scores, beam, dim=1
                    )

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + float(local_best_scores[0, j])
                    new_hyp["yseq"] = [0] * (1 + len(hyp["yseq"]))
                    new_hyp["yseq"][: len(hyp["yseq"])] = hyp["yseq"]
                    new_hyp["yseq"][len(hyp["yseq"])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp["rnnlm_prev"] = rnnlm_state
                    if lpz is not None:
                        new_hyp["ctc_state_prev"] = ctc_states[joint_best_ids[0, j]]
                        new_hyp["ctc_score_prev"] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x["score"], reverse=True
                )[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug("number of pruned hypothes: " + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    "best hypo: "
                    + "".join([char_list[int(x)] for x in hyps[0]["yseq"][1:]])
                )

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info("adding <eos> in the last postion in the loop")
                for hyp in hyps:
                    hyp["yseq"].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp["yseq"][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp["yseq"]) > minlen:
                        hyp["score"] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp["score"] += recog_args.lm_weight * rnnlm.final(
                                hyp["rnnlm_prev"]
                            )
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info("end detected at %d", i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug("remeined hypothes: " + str(len(hyps)))
            else:
                logging.info("no hypothesis. Finish decoding.")
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        "hypo: " + "".join([char_list[int(x)] for x in hyp["yseq"][1:]])
                    )

            logging.debug("number of ended hypothes: " + str(len(ended_hyps)))

        nbest_hyps = sorted(ended_hyps, key=lambda x: x["score"], reverse=True)[
            : min(len(ended_hyps), recog_args.nbest)
        ]

        # check number of hypotheis
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            # should copy becasuse Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info("total log probability: " + str(nbest_hyps[0]["score"]))
        logging.info(
            "normalized log probability: "
            + str(nbest_hyps[0]["score"] / len(nbest_hyps[0]["yseq"]))
        )
        return nbest_hyps

    def calculate_all_attentions(self, langs, xs_pad, ilens, ys_pad, langs_f):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights (B, H, Lmax, Tmax)
        :rtype: float ndarray
        """
        self.eval()
        with torch.no_grad():
            self.forward(langs, xs_pad, ilens, ys_pad, langs_f, '3p')
        ret = dict()
        for name, m in self.named_modules():
            if (
                isinstance(m, MultiHeadedAttention)
                or isinstance(m, DynamicConvolution)
                or isinstance(m, RelPositionMultiHeadedAttention)
            ):
                ret[name] = m.attn.cpu().numpy()
            if isinstance(m, DynamicConvolution2D):
                ret[name + "_time"] = m.attn_t.cpu().numpy()
                ret[name + "_freq"] = m.attn_f.cpu().numpy()
        self.train()
        return ret

    def calculate_all_ctc_probs(self,  langs, xs_pad, ilens, ys_pad, langs_f):
        """E2E CTC probability calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: CTC probability (B, Tmax, vocab)
        :rtype: float ndarray
        """
        ret = None
        if self.mtlalpha == 0:
            return ret

        self.eval()
        with torch.no_grad():
            self.forward(langs, xs_pad, ilens, ys_pad, langs_f, '3p')
        for name, m in self.named_modules():
            if 'ctc_h' in name and isinstance(m, CTC) and m.probs is not None:
                ret = m.probs.cpu().numpy()
        self.train()
        return ret
