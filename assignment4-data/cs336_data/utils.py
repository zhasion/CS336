import os
import resiliparse.parse
import resiliparse.extract.html2text
import fasttext
import regex as re
import mmh3
import unicodedata
import numpy as np
import shutil

from typing import Any
from nltk.tokenize import word_tokenize

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

identify_language_model = fasttext.load_model('data/lid.176.bin')
nsfw_model = fasttext.load_model('data/jigsaw_fasttext_bigrams_nsfw_final.bin')
toxic_speech_model = fasttext.load_model('data/jigsaw_fasttext_bigrams_hatespeech_final.bin')
quality_model = fasttext.load_model('data/quality_classifier/quality.bin')

MAIL_REGEX = re.compile(
    r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
)

PHONE_REGEX = re.compile(
    # r'(?<!\w)(?:\+1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)'
    r'(\()?\d{3}(?(1)\))[- ]?\d{3}[- ]?\d{4}'
)

IPV4_REGEX = re.compile(
    r'((25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)'
)

label_map = {
    "positive": "wiki",
    "negative": "cc",
}


def extract_text_from_html_bytes(html_bytes: bytes) -> str:
    content_type = resiliparse.parse.encoding.detect_encoding(html_bytes)
    content = html_bytes.decode(content_type, errors="ignore")
    return resiliparse.extract.html2text.extract_plain_text(content)


def identify_language(text: str) -> tuple[str, float]:
    predicts = identify_language_model.predict(text.replace('\n', ' '))

    language = predicts[0][0].replace('__label__', '')
    score = float(predicts[1][0])
    return language, score


def mask_email(text: str) -> tuple[str, int]:
    return MAIL_REGEX.subn('|||EMAIL_ADDRESS|||', text)


def mask_phone_numbers(text: str) -> tuple[str, int]:
    return PHONE_REGEX.subn('|||PHONE_NUMBER|||', text)


def mask_ip_address(text: str) -> tuple[str, int]:
    return IPV4_REGEX.subn('|||IP_ADDRESS|||', text)


def classify_nsfw(text: str) -> tuple[Any, float]:
    predicts = nsfw_model.predict(text.replace('\n', ' '))

    label = predicts[0][0].replace('__label__', '')
    score = float(predicts[1][0])

    return label, score


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    predicts = toxic_speech_model.predict(text.replace('\n', ' '))
    label = predicts[0][0].replace('__label__', '')
    score = float(predicts[1][0])

    return label, score


def classify_quality(text: str) -> tuple[Any, float]:
    labels, probs = quality_model.predict(text.replace("\n", " "), k=1)

    label = labels[0].replace("__label__", "")
    return label_map[label], probs[0]


def gopher_quality_filter(text: str) -> bool:
    MIN_NUM_WORDS = 50
    MAX_NUM_WORDS = 100_000
    MIN_NUM_MEAN_WORD_LENGTH = 3
    MAX_NUM_MEAN_WORD_LENGTH = 10
    MAX_RATIO_ELLIPSIS = 0.3
    MIN_RATIO_ALPHA_CHAR = 0.8
    
    tokens = word_tokenize(text)

    # Contain less than 50 or more than 100,000 words.
    token_len = len(tokens)
    if token_len > MAX_NUM_WORDS or token_len < MIN_NUM_WORDS:
        return False
    
    # Have a mean word length outside the range of 3 to 10 characters.
    mean_char = sum([len(x) for x in tokens]) / token_len
    if mean_char > MAX_NUM_MEAN_WORD_LENGTH or mean_char < MIN_NUM_MEAN_WORD_LENGTH:
        return False

    # Have more than 30% of lines ending with an ellipsis (“...”).
    line_list = text.splitlines()
    count = 0
    for line in line_list:
        if line.endswith('...'):
            count += 1
    if count / len(line_list) > MAX_RATIO_ELLIPSIS:
        return False
    
    # Contain less than 80% of words with at least one alphabetic character.
    ct_non_alpha_tokens = 0
    for token in tokens:
        if not any(x.isalpha for x in token):
            ct_non_alpha_tokens += 1
    if ct_non_alpha_tokens / token_len > MIN_RATIO_ALPHA_CHAR:
        return False
    
    return True


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    unique_lines = {}
    for file in input_files:
        file_name = os.path.basename(file)
        with open(file) as f:
            for line in f:
                line_hash = mmh3.hash(line, signed=False)
                if line_hash not in unique_lines:
                    unique_lines[line_hash] = {
                        "first_file": file_name,
                        "count": 0,
                    }
                unique_lines[line_hash]["count"] += 1

    for file in input_files:
        file_name = os.path.basename(file)
        with open(file) as f:
            with open(os.path.join(output_directory, file_name), "w") as f_out:
                for line in f:
                    line_hash = mmh3.hash(line, signed=False)
                    if unique_lines[line_hash]["count"] > 1 or unique_lines[line_hash]["first_file"] != file_name:
                        continue

                    f_out.write(line)



def normalize_text(s: str) -> str:
    # lowercasing.
    s = s.lower()
    # removing punctuation.
    s = re.sub(r'[^\w\s]', ' ', s)
    # normalizing whitespaces.
    s = re.sub(r'\s+', ' ', s)
    # removing accents.
    # appling NFD unicode normalization.
    s = unicodedata.normalize('NFD', s)

    return s


def get_minhash(ngram_set: list[str], num_hash_function: int):
    function_n = np.arange(num_hash_function)
    mins = np.full(num_hash_function, float('inf'))

    for ngram in ngram_set:
        h = mmh3.hash(ngram)
        mins = np.minimum(mins, h ^ function_n)
    return mins.tolist()


def get_file_normalized_ngram_set(file: os.PathLike, ngrams: int):
    with open(file, encoding="utf-8", errors="ignore") as f:
        text = f.read()

    words = text.split()
    return set(" ".join(words[i : i + ngrams]) for i in range(len(words) - ngrams + 1))


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    
    bands = dict()
    candiate_dup = set()
    ngram_set_dict = dict()
    cluster = dict()
    for path in input_files:
        ngram_set = get_file_normalized_ngram_set(path, ngrams)
        minhash = get_minhash(ngram_set, num_hashes)
        ngram_set_dict[path] = ngram_set

        for band in range(num_bands):
            band_minhash = tuple(minhash[band::num_bands])
            if band_minhash not in bands.keys():
                bands[band_minhash] = []
            bands[band_minhash].append(path)

            for item in bands[band_minhash]:
                if item == band_minhash:
                    continue
                candiate_dup.add((path, item))
    
    for path1, path2 in candiate_dup:
        set1 = ngram_set_dict[path1]
        set2 = ngram_set_dict[path2]

        jaccard_similarity = len(set1 & set2) / len(set1 | set2)

        if jaccard_similarity >= jaccard_threshold:
            cluster.setdefault(path1, set()).add(path2)
            cluster[path2] = cluster[path1]
    
    cluster_set = (frozenset(x) for x in cluster.values())

    save_path_list = [x for x in input_files if x not in cluster.keys()]
    save_path_list += [tuple(cluster)[0] for cluster in cluster_set]

    for path in save_path_list:
        save_path = os.path.join(output_directory, os.path.basename(path))
        shutil.copy(path, save_path)
            
