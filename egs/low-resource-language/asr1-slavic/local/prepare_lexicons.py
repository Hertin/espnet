#!/usr/bin/env python
import re
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path
from subprocess import run, PIPE, DEVNULL
from tempfile import NamedTemporaryFile

from typing import List, Dict

BABELCODE2LANG = {
    "101": "Cantonese",
    "102": "Assamese",
    "103": "Bengali",
    "104": "Pashto",
    "105": "Turkish",
    "106": "Tagalog",
    "107": "Vietnamese",
    "201": "Haitian",
    "202": "Swahili",
    "203": "Lao",
    "204": "Tamil",
    "205": "Kurmanji",
    "206": "Zulu",
    "207": "Tok-Pisin",
    "301": "Cebuano",
    "302": "Kazakh",
    "303": "Telugu",
    "304": "Lithuanian",
    "305": "Guarani",
    "306": "Igbo",
    "307": "Amharic",
    "401": "Mongolian",
    "402": "Javanese",
    "403": "Dholuo",
    "404": "Georgian",
    "505": "Dutch",
}


def remove_special_chars(tokens, token_special):
    tokens = [t.strip() for t in tokens if t.strip() not in ['<', '>', '-', '-', '─', '─', '──', '───', '□', '"', '(', ')', ',', '.', '/', '?', '&', '○']]
    ret_tokens = []
    for t in tokens:
        if '_' in t: # if token contains '_' or '-' (A_B-C)
            t = t.replace('_', ' ') # convert it to without-hypen version (A_B-C => ABC)
            token_special[t] = t
        if t.startswith('───'):
            token_converted = t.replace('───', '')
            token_special[token_converted] = t
        elif t.startswith('─'):
            token_converted = t.replace('─', '')
            token_special[token_converted] = t
        elif t.startswith('-'):
            token_converted = t.replace('-', '')
            token_special[token_converted] = t
        else:
            token_converted = t # no conversion
        ret_tokens.append(token_converted)
    return ret_tokens, token_special

def main():
    # noinspection PyTypeChecker
    parser = ArgumentParser(
        description="Prepare LanguageNet IPA phonetic transcripts for a given language and data directory.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-l", "--lang", help="Selected language.")
    parser.add_argument(
        "-d", "--data-dir", help="Path to Kaldi data directory with text file."
    )
    parser.add_argument(
        "-g",
        "--g2p-models-dir",
        default="g2ps/models",
        help="Directory with phonetisaurus g2p FST models for all languages.",
    )
    parser.add_argument(
        "-s",
        "--substitute-text",
        action="store_true",
        help="Will save original text in text.bkp and save the IPA transcript to text.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    g2p_models_dir = Path(args.g2p_models_dir)
    lang = BABELCODE2LANG.get(args.lang, args.lang).lower()
    lang2fst = G2PModelProvider(g2p_models_dir)
    model = lang2fst.get(lang)
    if not model:
        raise ValueError(f"No G2P FST model for language {lang} in {g2p_models_dir}")

    text = data_dir / "text"
    if not text.exists():
        raise ValueError(f"No such file: {text}")

    lexicon_path = data_dir / "lexicon_ipa.txt"
    
    token_special = {}
    with NamedTemporaryFile("w+") as f:
        text_bbkp = text.with_suffix(".bbkp")
        shutil.copyfile(text, text_bbkp)
        if lang == "mandarin":
            # use jieba to tokenize mandarin
            import jieba
            jieba.enable_paddle()
            with text.open("w") as fout, text_bbkp.open("r") as fin:
                for l in fin:
                    # remove "\n" and utterance id, and split into tokens 
                    utt, txt = l.strip().split(' ', maxsplit=1)
                    tokens = jieba.cut(txt, use_paddle=True)
                    tokens, token_special = remove_special_chars(tokens, token_special)
                    line = ' '.join(tokens)
                    if not line: continue
                    fout.write(f'{utt} {line}\n')
        elif lang == 'thai':
            from thai_segmenter import sentence_segment, tokenize
            with text.open("w") as fout, text_bbkp.open("r") as fin:
                for l in fin:
                    # remove "\n" and utterance id, and split into tokens 
                    utt, txt = l.strip().split(' ', maxsplit=1)
                    tokens = tokenize(txt)
                    tokens, token_special = remove_special_chars(tokens, token_special)
                    line = ' '.join(tokens)
                    if not line: continue
                    fout.write(f'{utt} {line}\n')
        else: # other languages; Assume already tokenized
            with text.open("w") as fout, text_bbkp.open("r") as fin:
                for l in fin:
                    # remove "\n" and utterance id, and split into tokens 
                    utt, *tokens = l.strip().split()
                    tokens, token_special = remove_special_chars(tokens, token_special)
                    line = ' '.join(tokens)
                    if not line: continue
                    fout.write(f'{utt} {line}\n')
        uniq_words = run(
            f"cut -f2- -d' ' {text} | tr ' ' '\n' | sort | uniq",
            text=True,
            check=True,
            shell=True,
            stdout=PIPE,
        ).stdout
        f.write(uniq_words)
        f.flush()

        lexicon_str = run(
            [
                "phonetisaurus-g2pfst",
                f"--model={g2p_models_dir / model}",
                f"--wordlist={f.name}",
            ],
            check=True,
            text=True,
            stdout=PIPE,
            stderr=DEVNULL,
        ).stdout

        lines = []
        # problematic_lines = []
        
        print('#'*100, lang, data_dir, 'special tokens')
        with open(data_dir / 'token_special.txt', 'w') as fsp:
            for to, torig in token_special.items():
                print(f'{to}:\t{torig}')
                fsp.write(f'{to}:\t{torig}\n')
            print('='*100)
            fsp.write(f'='*100+'\n')
            for i, l in enumerate(lexicon_str.strip().split('\n')):
                try:
                    token, score, phones = l.split('\t')
                except:
                    fsp.write(f'Exception at line {i} {l}')
                    print(i, 'l', l,  l.split('\t'))
                    continue
                if not phones: continue
                if token in token_special:
                    fsp.write(f'{token_special[token]} {l}')
                    print(f'{token_special[token]} {l}')

                # remove lines that are empty
                lines.append(l)
        lexicon_str = '\n'.join(lines) + '\n'

        print('@'*100, lang, data_dir, 'lexicon')
        print(lexicon_str[:1000])

        lexicon_path.write_text(lexicon_str)

        lexicon = LanguageNetLexicon.from_path(lexicon_path)

        text_bkp = text.with_suffix(".bkp")
        shutil.copyfile(text, text_bkp)
        text_ipa = text.with_suffix(".ipa")
        with text_bkp.open() as fin, text_ipa.open("w") as fout:
            for line in fin:
                utt_id, *words = line.strip().split()
                phonetic = ["".join(lexicon.transcribe(w)).strip() for w in words]
                if not phonetic or '' in phonetic:
                    continue  # skip empty utterances and utterances that have empty word
                print(utt_id, *[w for w in phonetic if w], file=fout)

        if args.substitute_text:
            shutil.copyfile(text_ipa, text)


class G2PModelProvider:
    def __init__(self, g2p_models_dir: Path):
        self.lang2fst = {
            line.split("_")[0]: line
            for line in run(
                ["ls", g2p_models_dir], text=True, check=True, stdout=PIPE
            ).stdout.split()
        }

    def get(self, lang: str) -> str:
        if lang == "arabic":
            lang = "gulf-arabic"  # TODO: confirm that GlobalPhone has Gulf Arabic
        if lang == "cantonese":
            lang = "yue"  # TODO: confirm that GlobalPhone has Gulf Arabic
        return self.lang2fst[lang]


Phone = str


class LanguageNetLexicon:
    WORD_SEPARATOR = "#"
    SYLLABLE_SEPARATOR = "."
    SPECIAL_TOKEN_RE = re.compile(r"<.+>")

    def __init__(self, lexicon: Dict[str, List[str]]):
        self.lexicon = lexicon

    @staticmethod
    def from_path(p: Path) -> "LanguageNetLexicon":
        lexicon = {}
        with p.open() as f:
            for line in f:
                word, score, *phones = line.strip().split()
                lexicon[word] = phones
        return LanguageNetLexicon(lexicon)

    def transcribe(
        self,
        word: str,
        strip_special_markers: bool = True,
        remove_special_tokens: bool = False,
    ) -> List[Phone]:
        # Treat special words as their own phones or remove
        if self.SPECIAL_TOKEN_RE.match(word) and (not word.startswith('<')) and (not word.endswith('->')):
            return [] if remove_special_tokens else [word]

        phonetic = self.lexicon.get(word.strip(), "")

        def is_not_special_marker_or_special_markers_are_ok(p: str) -> bool:
            if not strip_special_markers:
                return True
            return not any(
                p == sym for sym in [self.SYLLABLE_SEPARATOR, self.WORD_SEPARATOR]
            )

        phonetic = [
            p
            for p in phonetic
            if p and is_not_special_marker_or_special_markers_are_ok(p) and (not p.isdigit())
        ]
        return phonetic


if __name__ == "__main__":
    main()
