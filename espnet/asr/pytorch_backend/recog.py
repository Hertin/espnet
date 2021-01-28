"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import torch

from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.pytorch_backend.asr import load_trained_model
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.beam_search import BeamSearch
from espnet.nets.lm_interface import dynamic_import_lm
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.io_utils import LoadInputsAndTargets

from espnet.utils.dataset import ChainerDataLoader
from espnet.utils.dataset import TransformDataset
from espnet.utils.training.batchfy import make_batchset
from espnet.asr.pytorch_backend.asr import CustomConverter, _recursive_to
from ctcdecode import CTCBeamDecoder

def recog_v2(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog
        only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments.
        See py:func:`espnet.bin.asr_recog.get_parser` for details

    """
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.rnnlm:
        lm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        # NOTE: for a compatibility with less than 0.5.0 version models
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(train_args.char_list), lm_args)
        torch_load(args.rnnlm, lm)
        lm.eval()
    else:
        lm = None

    if args.ngram_model:
        from espnet.nets.scorers.ngram import NgramFullScorer
        from espnet.nets.scorers.ngram import NgramPartScorer

        if args.ngram_scorer == "full":
            ngram = NgramFullScorer(args.ngram_model, train_args.char_list)
        else:
            ngram = NgramPartScorer(args.ngram_model, train_args.char_list)
    else:
        ngram = None

    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["ngram"] = ngram
    scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
        ngram=args.ngram_weight,
        length_bonus=args.penalty,
    )
    beam_search = BeamSearch(
        beam_size=args.beam_size,
        vocab_size=len(train_args.char_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=train_args.char_list,
        pre_beam_score_key=None if args.ctc_weight == 1.0 else "full",
    )
    # TODO(karita): make all scorers batchfied
    if args.batchsize == 1:
        non_batch = [
            k
            for k, v in beam_search.full_scorers.items()
            if not isinstance(v, BatchScorerInterface)
        ]
        if len(non_batch) == 0:
            beam_search.__class__ = BatchBeamSearch
            logging.info("BatchBeamSearch implementation is selected.")
        else:
            logging.warning(
                f"As non-batch scorers {non_batch} are found, "
                f"fall back to non-batch implementation."
            )

    if args.ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()
    beam_search.to(device=device, dtype=dtype).eval()

    # read json data
    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]

    from collections import OrderedDict
    import random
    random.seed(args.seed)
    items = list(js.items())
    random.shuffle(items)
    js = OrderedDict(items[:2])
    logging.warning(f'data json len {len(js)}')

    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info("(%d/%d) decoding " + name, idx, len(js.keys()))
            batch = [(name, js[name])]
            feat = load_inputs_and_targets(batch)[0][0]
            enc = model.encode(torch.as_tensor(feat).to(device=device, dtype=dtype))
            nbest_hyps = beam_search(
                x=enc, maxlenratio=args.maxlenratio, minlenratio=args.minlenratio
            )
            nbest_hyps = [
                h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), args.nbest)]
            ]
            new_js[name] = add_results_to_json(
                js[name], nbest_hyps, train_args.char_list
            )

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )



def recog_ctconly(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog
        only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments.
        See py:func:`espnet.bin.asr_recog.get_parser` for details

    """

    logging.warning(f'RECOGCTCONLY')
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)

    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()

    # logging.warning(f'Recog deep [model.args] {model.args}')

    with open(args.recog_json, "rb") as f:
        recog_json = json.load(f)["utts"]

    use_sortagrad = model.args.sortagrad == -1 or model.args.sortagrad > 0

    converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)

    # make minibatch list (variable length)
    recog = make_batchset(
        recog_json,
        16, # model.args.batch_size,
        model.args.maxlen_in,
        model.args.maxlen_out,
        model.args.minibatches,
        min_batch_size=model.args.ngpu if model.args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=model.args.batch_count,
        batch_bins=400000, #model.args.batch_bins,
        batch_frames_in=model.args.batch_frames_in,
        batch_frames_out=model.args.batch_frames_out,
        batch_frames_inout=model.args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    load_rc = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=model.args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )

    recog_iter = ChainerDataLoader(
        dataset=TransformDataset(
            recog, lambda data: converter([load_rc(data)]), utt=True
        ),
        batch_size=1,
        num_workers=model.args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )

    logging.info(f'Character list: {model.args.char_list}')

    decoder = CTCBeamDecoder(
        labels=model.args.char_list, beam_width=args.beam_size, log_probs_input=True
    )

    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    for batch in recog_iter:
        names, x = batch[0], batch[1:]
        # logging.warning(f"Recog deep [names] {names}")
        x = _recursive_to(x, device)

        xs_pad, ilens, ys_pad = x
        logprobs, seq_lens = model.encode_with_length(xs_pad, ilens)

        # logging.warning(f'Recog Deep [logprobs] {logprobs.size()}')
        out, scores, offsets, seq_lens = decoder.decode(logprobs, seq_lens)
        for hyp, trn, length, name, score in zip(out, ys_pad, seq_lens, names, scores): # iterate batch
            # logging.warning(f'{score}')
            best_hyp = hyp[0,:length[0]]

            new_js[name] = add_results_to_json(
                js[name], [{"yseq": best_hyp, "score": float(score[0])}], model.args.char_list
            )


            # logging.warning(f'Recog deep [new_js] {new_js}')
            # break

        # raise
    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )


def recog_ctconly_lang(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog
        only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments.
        See py:func:`espnet.bin.asr_recog.get_parser` for details

    """

    logging.warning(f'RECOGCTCONLYLANG')
    logging.warning(f'all_langs {args.train_langs}')
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)

    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()

    # logging.warning(f'Recog deep [model.args] {model.args}')

    with open(args.recog_json, "rb") as f:
        recog_json = json.load(f)["utts"]

    use_sortagrad = model.args.sortagrad == -1 or model.args.sortagrad > 0

    converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)

    # make minibatch list (variable length)
    recog = make_batchset(
        recog_json,
        16, # model.args.batch_size,
        model.args.maxlen_in,
        model.args.maxlen_out,
        model.args.minibatches,
        min_batch_size=model.args.ngpu if model.args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=model.args.batch_count,
        batch_bins=400000, #model.args.batch_bins,
        batch_frames_in=model.args.batch_frames_in,
        batch_frames_out=model.args.batch_frames_out,
        batch_frames_inout=model.args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    load_rc = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=model.args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )

    recog_iter = ChainerDataLoader(
        dataset=TransformDataset(
            recog, lambda data: converter([load_rc(data)]), utt=True, lang_onehot=True, all_lang=args.train_langs
        ),
        batch_size=1,
        num_workers=model.args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )
    

    logging.info(f'Character list: {model.args.char_list}')

    decoder = CTCBeamDecoder(
        labels=model.args.char_list, beam_width=args.beam_size, log_probs_input=True
    )

    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    for batch in recog_iter:
        names, x = batch[0], batch[1:]
        
        # logging.warning(f"Recog deep [names] {names}")
        x = _recursive_to(x, device)

        langs, xs_pad, ilens, ys_pad = x
        logging.warning(f'parameters, names {names}')
        logging.warning(f'parameters, langs {langs}')
        # logging.warning(f'parameters, names {names}')
        logprobs, seq_lens = model.encode_with_length(langs, xs_pad, ilens)

        # logging.warning(f'Recog Deep [logprobs] {logprobs.size()}')
        out, scores, offsets, seq_lens = decoder.decode(logprobs, seq_lens)
        for hyp, trn, length, name, score in zip(out, ys_pad, seq_lens, names, scores): # iterate batch
            # logging.warning(f'{score}')
            best_hyp = hyp[0,:length[0]]

            new_js[name] = add_results_to_json(
                js[name], [{"yseq": best_hyp, "score": float(score[0])}], model.args.char_list
            )


            # logging.warning(f'Recog deep [new_js] {new_js}')
            # break

        # raise
    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )


def recog_seg(args):
    """Decode with custom models that implements ScorerInterface.

    Notes:
        The previous backend espnet.asr.pytorch_backend.asr.recog
        only supports E2E and RNNLM

    Args:
        args (namespace): The program arguments.
        See py:func:`espnet.bin.asr_recog.get_parser` for details

    """

    logging.warning(f'RECOGSEG')
    logging.warning("experimental API for custom LMs is selected by --api v2")
    if args.batchsize > 1:
        raise NotImplementedError("multi-utt batch decoding is not implemented")
    if args.streaming_mode is not None:
        raise NotImplementedError("streaming mode is not implemented")
    if args.word_rnnlm:
        raise NotImplementedError("word LM is not implemented")

    set_deterministic_pytorch(args)
    model, train_args = load_trained_model(args.model)

    assert isinstance(model, ASRInterface)
    model.eval()

    load_inputs_and_targets = LoadInputsAndTargets(
        mode="asr",
        load_output=False,
        sort_in_input_length=False,
        preprocess_conf=train_args.preprocess_conf
        if args.preprocess_conf is None
        else args.preprocess_conf,
        preprocess_args={"train": False},
    )

    if args.ngpu == 1:
        device = "cuda"
    else:
        device = "cpu"
    dtype = getattr(torch, args.dtype)
    logging.info(f"Decoding device={device}, dtype={dtype}")
    model.to(device=device, dtype=dtype).eval()

    # logging.warning(f'Recog deep [model.args] {model.args}')

    with open(args.recog_json, "rb") as f:
        recog_json = json.load(f)["utts"]

    use_sortagrad = model.args.sortagrad == -1 or model.args.sortagrad > 0

    converter = CustomConverter(subsampling_factor=model.subsample[0], dtype=dtype)

    # make minibatch list (variable length)
    recog = make_batchset(
        recog_json,
        16, # model.args.batch_size,
        model.args.maxlen_in,
        model.args.maxlen_out,
        model.args.minibatches,
        min_batch_size=model.args.ngpu if model.args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=model.args.batch_count,
        batch_bins=400000, #model.args.batch_bins,
        batch_frames_in=model.args.batch_frames_in,
        batch_frames_out=model.args.batch_frames_out,
        batch_frames_inout=model.args.batch_frames_inout,
        iaxis=0,
        oaxis=0,
    )
    load_rc = LoadInputsAndTargets(
        mode="asr",
        load_output=True,
        preprocess_conf=model.args.preprocess_conf,
        preprocess_args={"train": False},  # Switch the mode of preprocessing
    )

    recog_iter = ChainerDataLoader(
        dataset=TransformDataset(
            recog, lambda data: converter([load_rc(data)]), utt=True
        ),
        batch_size=1,
        num_workers=model.args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )

    logging.info(f'Character list: {model.args.char_list}')

    decoder = CTCBeamDecoder(
        labels=model.args.char_list, beam_width=args.beam_size, log_probs_input=True
    )

    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    for batch in recog_iter:
        names, x = batch[0], batch[1:]
        # logging.warning(f"Recog deep [names] {names}")
        x = _recursive_to(x, device)

        xs_pad, ilens, ys_pad = x
        logits = model.encode(xs_pad, ilens)
        # logging.warning(f"Recog logit {logits.size()}")
        # torch.argmax(logit.view())
        predicts = torch.argmax(logits, dim=1)
        # logging.warning(f"Recog logit {logits.size()} {predicts.size()}")
        # logging.warning(f"Recog logit {predicts[:10]}")
        # logging.warning(f"Recog logit {ys_pad[:10]}")
        for pred, trn, name, logit in zip(predicts, ys_pad, names, logits):
            best_hyp = pred.view(-1)
            # logging.warning(f'{torch.nn.functional.pad(best_hyp, (1,0))}, {model.args.char_list[pred]}')
            new_js[name] = add_results_to_json(
                js[name], [{"yseq": torch.nn.functional.pad(best_hyp, (1,0)), "score": float(logit[best_hyp])}], model.args.char_list
            )

            # logging.warning(f'{new_js[name]}')

    with open(args.result_label, "wb") as f:
        f.write(
            json.dumps(
                {"utts": new_js}, indent=4, ensure_ascii=False, sort_keys=True
            ).encode("utf_8")
        )
