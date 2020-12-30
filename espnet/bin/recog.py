"""V2 backend for `asr_recog.py` using py:class:`espnet.nets.beam_search.BeamSearch`."""

import json
import logging

import torch
import numpy as np
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
from espnet.utils.dataset import TransformDatasetEarEval
from espnet.utils.training.batchfy import make_batchset
from espnet.asr.pytorch_backend.asr_ear import CustomConverter, _recursive_to
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
    logging.warning(f'Recog [ctc weight] {args.ctc_weight}')
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

    scorers = model.scorers()
    scorers["lm"] = lm
    scorers["length_bonus"] = LengthBonus(len(train_args.char_list))
    weights = dict(
        decoder=1.0 - args.ctc_weight,
        ctc=args.ctc_weight,
        lm=args.lm_weight,
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
        pre_beam_score_key=None if args.ctc_weight == 1.0 else "decoder",
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




def recog_deepspeech(args):
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
    # logging.warning(f'Recog [model signature map] {model.signature_map.requires_grad} {model.signature_map}')
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
        model.args.batch_size,
        model.args.maxlen_in,
        model.args.maxlen_out,
        model.args.minibatches,
        min_batch_size=model.args.ngpu if model.args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=model.args.batch_count,
        batch_bins=1200000, # model.args.batch_bins,
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
        dataset=TransformDatasetEarEval(recog, lambda data: converter([load_rc(data)])),
        batch_size=1,
        num_workers=1,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )

    # dump/eval_203/deltafalse/split1utt/data.1.json
    lang = args.recog_json.strip('dump/').split('/')[0].split('_')[-1] # get language
    logging.warning(f'recog [lang] {lang}')
    # with open(f'data/lang_1char/train_units_{lang}_omit.json', 'r') as f:
    #     char_list = json.load(f)
    # signature_map = np.load(f'data/arti_attr_mat_{lang}_unseenFalse.npy')
    # signature_map = np.load(f'data/arti_attr_mat_{lang}_omit.npy')
    # signature_map = model.signature_map.detach().cpu().numpy()
    char_list = model.args.char_list
    # prob_adjust = torch.from_numpy(np.load(f'data/probs_{lang}_omit.npy')).float().to(device)
    
    # logging.warning(f'signature map {model.signature_map.size()} {model.signature_map[:,0]} {model.signature_map}')
    # model.signature_map = torch.from_numpy(signature_map).float().to(device)
    # logging.warning(f'size {model.signature_map.size()} {prob_adjust.size()}')
    # logging.warning(f'size {model.signature_map.size()} {torch.diag(prob_adjust).size()}')
    # model.ctc.signature_map = torch.matmul(model.signature_map, torch.diag(prob_adjust))
    
    # logging.warning(f'recog [signature_map] {signature_map.shape}')
    logging.warning(f'{model.args.char_list}')
    logging.warning(f'{len(char_list)}')
    decoder = CTCBeamDecoder(
        labels=char_list, beam_width=args.beam_size, log_probs_input=True
    )

    with open(args.recog_json, "rb") as f:
        js = json.load(f)["utts"]
    new_js = {}

    for batch in recog_iter:
        names, x = batch[0], batch[1:]
        
        logging.warning(f"Recog deep [names] {names}")
        
        x = _recursive_to(x, device)

        
        xs_pad, ilens, ys_pad = x
        logprobs, seq_lens = model.encode(xs_pad, ilens)
        # probs = logprobs.exp() # B x T x D


        # prob_adjust.sum(dim=-1) # B x T
        # prob_adjust[prob_adjust!=0].sum(dim=-1) # B x T



        logging.warning(f'Recog Deep [logprobs] {logprobs.size()}')
        out, scores, offsets, seq_lens = decoder.decode(logprobs, seq_lens)
        for hyp, trn, length, name in zip(out, ys_pad, seq_lens, names): # iterate batch

            best_hyp = hyp[0,:length[0]]

            new_js[name] = add_results_to_json(
                js[name], [{"yseq": best_hyp, "score": 0.0}], char_list
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


def recog_ctc(args):
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
    # logging.warning(f'Recog [model signature map] {model.signature_map.requires_grad} {model.signature_map}')
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
        dataset=TransformDatasetEarEval(recog, lambda data: converter([load_rc(data)])),
        batch_size=1,
        num_workers=model.args.n_iter_processes,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )
    
    # logging.warning(f'signature map {model.signature_map.size()} {model.signature_map[:,0]} {model.signature_map}')
    # model.signature_map = torch.from_numpy(signature_map).float().to(device)
    # logging.warning(f'size {model.signature_map.size()} {prob_adjust.size()}')
    # logging.warning(f'size {model.signature_map.size()} {torch.diag(prob_adjust).size()}')
    # model.ctc.signature_map = torch.matmul(model.signature_map, torch.diag(prob_adjust))
    
    # logging.warning(f'recog [signature_map] {signature_map.shape}')
    logging.warning(f'{model.args.char_list}')
    # logging.warning(f'{char_list}')
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
        logprobs, seq_lens = model.encode(xs_pad, ilens)
        # probs = logprobs.exp() # B x T x D

        # logging.warning(f'Recog Deep [logprobs] {logprobs.size()}')
        out, scores, offsets, seq_lens = decoder.decode(logprobs, seq_lens)
        for hyp, trn, length, name in zip(out, ys_pad, seq_lens, names): # iterate batch

            best_hyp = hyp[0,:length[0]]

            new_js[name] = add_results_to_json(
                js[name], [{"yseq": best_hyp, "score": 0.0}], model.args.char_list
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

def recog_transformer(args):
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
    # logging.warning(f'Recog [model signature map] {model.signature_map.requires_grad} {model.signature_map}')
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
        4, # model.args.batch_size,
        model.args.maxlen_in,
        model.args.maxlen_out,
        model.args.minibatches,
        min_batch_size=model.args.ngpu if model.args.ngpu > 1 else 1,
        shortest_first=use_sortagrad,
        count=model.args.batch_count,
        batch_bins=50000, #model.args.batch_bins,
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
        dataset=TransformDatasetEarEval(recog, lambda data: converter([load_rc(data)])),
        batch_size=1,
        num_workers=1,
        shuffle=not use_sortagrad,
        collate_fn=lambda x: x[0],
    )
    
    # logging.warning(f'signature map {model.signature_map.size()} {model.signature_map[:,0]} {model.signature_map}')
    # model.signature_map = torch.from_numpy(signature_map).float().to(device)
    # logging.warning(f'size {model.signature_map.size()} {prob_adjust.size()}')
    # logging.warning(f'size {model.signature_map.size()} {torch.diag(prob_adjust).size()}')
    # model.ctc.signature_map = torch.matmul(model.signature_map, torch.diag(prob_adjust))
    
    # logging.warning(f'recog [signature_map] {signature_map.shape}')
    logging.warning(f'{model.args.char_list}')
    # logging.warning(f'{char_list}')
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
        if xs_pad.size(1) >= 3500:
            logging.warning(f'Recog Transformer [x] {xs_pad.size()} skip')
            continue
        # logging.warning(f'Recog Deep [x] {xs_pad.size()}')
        logprobs, seq_lens = model.encode_with_length(xs_pad, ilens)
        # probs = logprobs.exp() # B x T x D
        
        # logging.warning(f'Recog Deep [logprobs] {logprobs.size()}')
        out, scores, offsets, seq_lens = decoder.decode(logprobs, seq_lens)
        for hyp, trn, length, name in zip(out, ys_pad, seq_lens, names): # iterate batch

            best_hyp = hyp[0,:length[0]]

            new_js[name] = add_results_to_json(
                js[name], [{"yseq": best_hyp, "score": 0.0}], model.args.char_list
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
