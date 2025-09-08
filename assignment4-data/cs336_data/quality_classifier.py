import os
import gzip
import random
from utils import *
from fastwarc.warc import ArchiveIterator, WarcRecordType


def get_positive_sample(file_path, max_num_smaple, save_dir):
    count = 0
    save_url_list = []
    with gzip.open(file_path, 'rt') as f:
        for line in f.readlines():
            line = line.strip()

            if not line:
                continue

            count += 1
            save_url_list.append(line)

            if count >= max_num_smaple:
                break
    
    with open(os.path.join(save_dir, 'positive_sample_url.txt'), 'w', encoding='utf8') as f:
        for each in save_url_list:
            f.write(f'{each}\n')


def download_url_content(save_dir):
    os.system(
        f"""
        wget --timeout=5 --tries=1 \
        -i {os.path.join(save_dir, 'positive_sample_url.txt')} \
        --warc-file={os.path.join(save_dir, 'unfiltered_positive_samples')} \
        -O /dev/null
        """
    )


def filter_positive_sample(save_dir):
    warc_path = os.path.join(save_dir, 'unfiltered_positive_samples.warc.gz')
    training_samples = []

    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if not record.record_type == WarcRecordType.response:
                continue
            
            if record.content_length == 0:
                continue
            
            if record.http_content_type is None:
                continue

            if  not record.http_content_type.startswith("text/html"):
                continue

            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            if not text.strip():
                continue

            nsfw_label, nsfw_conf = classify_nsfw(text)
            if nsfw_label == "nsfw" or (nsfw_label == "non-nsfw" and nsfw_conf < 0.9):
                continue

            toxic_label, toxic_conf = classify_toxic_speech(text)
            if toxic_label == "toxic" or (toxic_label == "non-toxic" and toxic_conf < 0.9):
                continue

            gopher_quality = gopher_quality_filter(text)
            if not gopher_quality:
                continue

            joined_text = text.replace("\n", " ")
            training_sample = f"__label__positive {joined_text}\n"
            training_samples.append(training_sample)

    output_path = os.path.join(save_dir, 'positive_sample.txt')
    with open(output_path, "w") as f:
        f.writelines(training_samples)


def get_negative_sample(warc_path, num_sample, save_dir):
    training_samples = []
    count = 0
    with open(warc_path, 'rb') as f:
        for record in ArchiveIterator(f):
            if not record.record_type == WarcRecordType.response:
                continue
            if not record.http_content_type.startswith("text/html"):
                continue

            html_bytes = record.reader.read()
            text = extract_text_from_html_bytes(html_bytes)

            if not text.strip():
                continue
            
            count += 1
            joined_text = text.replace("\n", " ")
            training_sample = f"__label__negative {joined_text}\n"
            training_samples.append(training_sample)
            if count >= num_sample:
                break

    output_path = os.path.join(save_dir, 'negative_sample.txt')
    with open(output_path, "w") as f:
        f.writelines(training_samples)


def train_model(save_dir):
    model = fasttext.train_supervised(
        input=os.path.join(save_dir, "quality.train"),
        epoch=30,
        lr=0.2,
    )
    model.save_model(os.path.join(save_dir, "quality.bin"))
    print(model.test(os.path.join(save_dir, "quality.valid"), k=1))


def merge_split_dataset(save_dir, valid_ratio = 0.2):
    dataset = []
    with open(os.path.join(save_dir, 'positive_sample.txt'), 'r') as f:
        dataset = f.readlines()
    
    with open(os.path.join(save_dir, 'negative_sample.txt'), 'r') as f:
        dataset += f.readlines()
    
    valid_num = int(valid_ratio * len(dataset))
    random.shuffle(dataset)

    with open(os.path.join(save_dir, 'quality.valid'), 'w') as f:
        for item in dataset[:valid_num]:
            f.writelines(item)

    with open(os.path.join(save_dir, 'quality.train'), 'w') as f:
        for item in dataset[valid_num:]:
            f.writelines(item)

def main():
    positive_file_path = 'data/enwiki-20240420-extracted_urls.txt.gz'
    warc_path = 'data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    save_dir = 'data/quality_classifier'
    max_num_positive_smaple = 1000
    max_num_negative_smaple = 1000

    os.makedirs(save_dir, exist_ok=True)

    # print('[get_positive_sample]')
    # get_positive_sample(positive_file_path, max_num_positive_smaple, save_dir)

    # print('[download_url_content]')
    # download_url_content(save_dir)

    print('[filter_positive_sample]')
    filter_positive_sample(save_dir)

    print('[get_negative_sample]')
    get_negative_sample(warc_path, max_num_negative_smaple, save_dir)

    print('[merge_split_dataset]')
    merge_split_dataset(save_dir)

    print('[train_model]')
    train_model(save_dir)


if __name__ == '__main__':
    main()