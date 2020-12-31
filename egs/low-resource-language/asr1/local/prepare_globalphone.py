#!/usr/bin/env python

# Copyright 2020 Johns Hopkins University (Piotr Żelasko)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import re
import subprocess
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from pathlib import Path
from sys import stdout
from typing import NamedTuple, Dict, Optional, List

from tqdm import tqdm

logger = logging.getLogger("prepare_globalphone")


def main():
    # noinspection PyTypeChecker
    parser = ArgumentParser(
        description="Prepare train/dev/eval splits for various GlobalPhone languages.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gp-path",
        required=True,
        help="Path to GlobalPhone directory with each language"
        " in a subdirectory with its corresponding codename; "
        'e.g. on JHU grid, "/export/corpora5/GlobalPhone" has'
        "subdirectories like S0192 (Arabic) or S0196 (Czech).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/GlobalPhone",
        required=True,
        help="Output root with data directories for each language and split.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        required=True,
        default="Arabic Czech Croatian French Korean Mandarin Spanish Thai Bulgarian Hausa Polish German Turkish Portuguese".split(),
        help="Which languages to prepare.",
    )
    parser.add_argument(
        "--romanized",
        action="store_true",
        help='Use "rmn" directories in GlobalPhone transcripts if available.',
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Turn detailed logging on for debugging.",
    )
    args = parser.parse_args()
    gp_path = Path(args.gp_path)
    data_dir = Path(args.output_dir)
    train_languages = args.languages
    romanized = args.romanized
    if args.verbose:
        fancy_logging()

    data_dir.mkdir(parents=True, exist_ok=True)
    assert gp_path.exists()

    for lang in tqdm(train_languages, desc="Preparing per-language data dirs"):
        logging.debug(f"Preparing language {lang}")
        # We need the following files
        #   text  utt2spk  wav.scp  segments  spk2utt
        # We'll create the last two outside of the script
        dataset = parse_gp(
            gp_path / LANG2CODE[lang] / lang, romanized=romanized, lang=lang
        )
        dataset.write(data_dir)


CODE2LANG = {
    "S0192": "Arabic",
    "S0195": "Croatian",
    "S0196": "Czech",
    "S0197": "French",
    "S0198": "German",
    "S0200": "Korean",
    "S0201": "Portuguese",
    "S0206": "Turkish",
    "S0193": "Mandarin",
    "S0203": "Spanish",
    "S0321": "Thai",
    "S0319": "Bulgarian",
    "S0320": "Polish",
    "S0347":"Hausa"
}

LANG2CODE = {l: c for c, l in CODE2LANG.items()}

LANG2ENCODING = {
    "Arabic": "ISO8859-1",
    "Bulgarian": "UTF-8",
    "Czech": "ISO8859-2",
    "Croatian": "ISO8859-2",
    "French": "ISO8859-2",
    "German": "ISO8859-1",
    "Hausa": "UTF-8",
    "Korean": "EUC-KR",
    "Mandarin": "GB18030",
    "Portuguese": "ISO8859-1",
    "Polish": "UTF-8",
    "Spanish": "ISO8859-1",
    "Thai": "TIS-620",
    "Turkish": "ISO8859-9"
}

LANG2SPLIT = {
    "Arabic": {
        "dev": [5, 36, 107, 164]
        + [
            1,
            2,
            3,
            4,
            6,
            7,
        ],  # TODO: Docs say: +6 TBA; I'm putting extra six speakers here
        "eval": [27, 39, 108, 137]
        + [8, 9, 10, 11, 12, 13],  # TODO: + 6 TBA; I'm putting extra six speakers here
    },
    "Czech": {  # TODO: docs say TBA... I'm putting 9 speakers in each set
        "dev": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "eval": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    },
    "Croatian": {  # as docs
        "eval": [37,38,39,40,41,42,43,44,45,47],
        "dev": [33,34,35,36,46,48,51,53,54,57],
    },
    "German": {
        "eval": [18, 20, 21, 26, 29, 73],
        "dev": [1,2,3,4,8,10],
    },
    "French": {
        "dev": [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
        ],  # TODO: no dev specified in docs; I'm putting eight speakers here
        "eval": list(range(91, 99)),  # 91-98
    },
    "Korean": {
        "dev": [6, 12, 25, 40, 45, 61, 84, 86, 91, 98],
        "eval": [19, 29, 32, 42, 51, 64, 69, 80, 82, 88],
    },
    "Mandarin": {
        "dev": list(range(28, 33)) + list(range(39, 45)),  # 28-32, 39-44
        "eval": list(range(80, 90)),  # 80-89
    },
    "Spanish": {
        "dev": list(range(1, 11)),  # 1-10
        "eval": list(range(11, 19)),  # 11-18
    },
    "Polish": {  # TODO: docs say TBA... I'm putting 9 speakers in each set
        "dev": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "eval": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    },
    "Bulgarian": {
        "eval": [40,59,63,68,95,109,110],
        "dev": [51,55,58,84,90,100,106],
    },
    "Hausa": {
        "eval": [2, 14, 25, 28, 30, 52, 53, 62, 70, 88],
        "dev": [18, 31, 34, 38, 46, 47, 50, 55,58, 72],
    },
    "Turkish": {
        "eval": [25, 30, 31, 32, 37, 39, 41, 46, 56, 63],
        "dev": [1,2,3,5,6,8,13,14,15,16,19],
    },
    "Thai": {
        "dev": [23, 25, 28, 37, 45, 61, 73, 85],
        "eval": list(range(101, 109)),  # 101-108
    },
    "Portuguese": {
        "dev": [64,65,72,73,102,103,104,132,133,134],
        "eval": [135,136,137,138,139,142,143,312],
    },
}


class Segment(NamedTuple):
    recording_id: str
    start_seconds: float
    end_seconds: float


KaldiTable = Dict[str, str]


class DataDir(NamedTuple):
    wav_scp: KaldiTable
    text: KaldiTable
    utt2spk: KaldiTable

    def write(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "wav.scp", "w") as f:
            for k, v in self.wav_scp.items():
                f.write(k + " " + v + "\n")
        with open(data_dir / "text", "w") as f:
            for k, v in self.text.items():
                f.write(k + " " + v + "\n")
        with open(data_dir / "utt2spk", "w") as f:
            for k, v in self.utt2spk.items():
                f.write(k + " " + v + "\n")


class GpDataset(NamedTuple):
    train: DataDir
    dev: DataDir
    eval: DataDir
    lang: str

    def write(self, data_root: Path):
        self.train.write(data_root / f"gp_{self.lang}_train")
        self.dev.write(data_root / f"gp_{self.lang}_dev")
        self.eval.write(data_root / f"gp_{self.lang}_eval")


def parse_gp(path: Path, lang: str, romanized=True):
    wav_scp: KaldiTable = create_wav_scp(path, lang)
    transcript_paths = find_transcripts(path, lang, romanized)

    # NOTE: We are using bytes instead of str because some GP transcripts have non-Unicode symbols which fail to parse

    lang_short = next(iter(wav_scp.keys()))[:2]
    text: KaldiTable = {}
    utt2spk: KaldiTable = {}
    # "None" makes it easier to pinpoint the error in debug when we fail to parse some file
    utt_id: Optional[str] = None  # e.g. AR059_UTT003
    spk_id: Optional[str] = None  # e.g. AR059
    num_utts = defaultdict(int)
    for p in sorted(transcript_paths):
        for line in read_as_utf8(p, lang=lang):
            m = re.match(
                r";SprecherID .*?(\d+)", line, flags=re.I
            )  # case-independent because "SpReChErId"...
            if m is not None:
                spk_id = f"{lang_short}{int(m.group(1)):03d}"
                continue
            m = re.match(r"; (\d+):", line)
            if m is not None:
                utt_id = f"{spk_id}_UTT{int(m.group(1)):03d}"
                continue
            assert spk_id is not None, f"No speaker ID at line {line}"
            assert utt_id is not None, f"No utterance ID at line {line}"
            # TODO: likely language-dependent text normalization would be adequate before adding transcript to text
            transcript = remove_special_symbols(line)
            if not transcript:
                logger.warning(f"Empty utterance {utt_id} in file {p}")
                continue
            text[utt_id] = transcript
            utt2spk[utt_id] = spk_id
            num_utts[spk_id] += 1

    run_utterance_diagnostics(lang, num_utts, text, utt2spk, wav_scp)

    def select(table: KaldiTable, split: str):
        if split == "train":
            selected_ids = {
                utt_id
                for utt_id in text
                if all(
                    number_of(utt_id) not in LANG2SPLIT[lang][split_]
                    for split_ in ("dev", "eval")
                )
            }
        else:
            selected_ids = {
                utt_id
                for utt_id in text
                if number_of(utt_id) in LANG2SPLIT[lang][split]
            }
        subset = {k: v for k, v in table.items() if k in selected_ids}
        # assert all(k in subset for k in selected_ids)
        return subset

    return GpDataset(
        train=DataDir(
            wav_scp=select(wav_scp, "train"),
            utt2spk=select(utt2spk, "train"),
            text=select(text, "train"),
        ),
        dev=DataDir(
            wav_scp=select(wav_scp, "dev"),
            utt2spk=select(utt2spk, "dev"),
            text=select(text, "dev"),
        ),
        eval=DataDir(
            wav_scp=select(wav_scp, "eval"),
            utt2spk=select(utt2spk, "eval"),
            text=select(text, "eval"),
        ),
        lang=lang,
    )


def find_transcripts(path: Path, lang: str, romanized: bool):
    tr_sfx = "rmn" if romanized else "trl"
    transcript_paths = list((path / tr_sfx).rglob(f"*.{tr_sfx}"))
    # If nothing found and romanized version was requested, try to fall back to non-romanized
    if not transcript_paths and tr_sfx == "rmn":
        tr_sfx = "trl"
        transcript_paths = list((path / tr_sfx).rglob(f"*.{tr_sfx}"))
        if not transcript_paths:
            raise ValueError(
                f"No transcripts found for {lang}! "
                f"(looking for extensions (rmn,trl) in: {path}/(rmn,trl))"
            )
    logging.debug(f"There are {len(transcript_paths)} transcript files")
    return transcript_paths


def read_as_utf8(path: Path, lang: str) -> List[str]:
    try:
        return subprocess.run(
            f"iconv -f {LANG2ENCODING[lang]} -t UTF-8 {path}",
            text=True,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
    except Exception as e:
        logger.error(
            f'Could not process file {path} with "{type(e).__name__}" "{e}" - skipping'
        )
        return []


def number_of(utt_id):
    try:
        return int(utt_id[2:5])
    except:
        logging.error(f"Can't extract the number of utterance id {utt_id}")
        raise


def run_utterance_diagnostics(
    lang: str,
    num_utts: Dict[str, int],
    text: KaldiTable,
    utt2spk: KaldiTable,
    wav_scp: KaldiTable,
):
    logging.debug(f"There is a total of {sum(num_utts.values())} utterances.")
    no_utt_speakers = [u for u, n in num_utts.items() if n == 0]
    if no_utt_speakers:
        logging.warning(
            f"There are {len(no_utt_speakers)}"
            f" speakers with 0 utterances in language {lang}."
        )
    missing_recordings = set(wav_scp).difference(utt2spk)
    if missing_recordings:
        logging.warning(
            f"There are {len(missing_recordings)} missing {lang} utterance IDs out of {len(wav_scp)} total "
            f"in wav.scp (use -v for details)"
        )
        logging.debug(
            f'The following utterance IDs are missing in wav.scp: {" ".join(sorted(missing_recordings))}'
        )
    missing_transcripts = set(utt2spk).difference(wav_scp)
    if missing_transcripts:
        logging.warning(
            f"There are {len(missing_transcripts)} missing {lang} utterance IDs out of {len(text)} total "
            f"in text and utt2spk (use -v for details)"
        )
        logging.debug(
            f'The following utterance IDs are missing in text and utt2spk: {" ".join(sorted(missing_transcripts))}'
        )


def create_wav_scp(path: Path, lang: str) -> KaldiTable:
    def make_id(path: Path) -> str:
        try:
            stem = path.stem
            if stem.endswith("adc."):
                stem = stem[:-4] + ".adc"
            id = stem.split(".")[0]
            parts = id.split("_")
            return f"{parts[0]}_UTT{int(parts[1]):03d}"
        except:
            logging.error(f"Can't parse from {path}")
            raise

    audio_paths = list((path / "adc").rglob("*.shn"))
    if not audio_paths:
        raise ValueError(
            f"No recordings found for {lang}! "
            f"(looking for extension \".shn\" here: {path / 'adc'})"
        )
    wav_scp = {make_id(p): decompressed(p) for p in sorted(audio_paths)}
    logging.debug(f"There are {len(wav_scp)} audio files")
    return wav_scp


def fancy_logging(level=logging.DEBUG, stream=stdout):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
        stream=stream,
    )


def decompressed(path: Path) -> str:
    return f"shorten -x {path} - | sox -t raw -r 16000 -b 16 -e signed-integer - -t wav - |"


ANY_WHITESPACE = re.compile(r"\s", re.UNICODE)


def remove_special_symbols(utt: str) -> str:
    # Remove begin-of-sentence and end-of-sentence
    # TODO: I don't know why these symbols appear in GP transcripts, we should find that out eventually
    utt = utt.replace("<s>", "").replace("</s>", "")
    # Remove Unicode fancy whitespaces
    utt = ANY_WHITESPACE.sub(" ", utt)
    return utt


if __name__ == "__main__":
    main()
