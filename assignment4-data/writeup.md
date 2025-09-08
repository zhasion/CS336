# CS336 Assignment 4 (data): Filtering Language Modeling Data
[TOC]



## 2 Filtering Common Crawl

### Problem (look_at_cc): 4 points

(a) Download the WARC file above, or find the copy we provide on the cluster. Let’s look at the first page in this file. This is a gzipped file, and you can browse its contents with:

```shell
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/ warc/CC-MAIN-20250417135010-20250417165010-00065.warc.gz
zcat /data/CC/example.warc.gz | less
```

less lets you browse the file using keyboard arrows, Page Up, Page Down. To exit, press “q”.

Look at the very first web page. What is its URL? Is it still accessible? Can you tell what the page seems to be about by looking at the raw HTML?

**Deliverable:** WARC-Target-URI: http://0371rykj.com/ipfhsb/34.html. It is not accessible. It might be the website of a Chinese company.

(b) Let’s now look at the corresponding WET file:

```shell
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-18/segments/1744889135610.12/ wet/CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
zcat /data/CC/example.warc.wet.gz | less
```

Note that the WET files contain HTTP headers (e.g., Content-Length) that are not part of the extracted text contents. If you look at the first example, you will see that it contains text that was extracted from the raw HTML you just saw.

Notice that much of the extracted text is reminiscent of the HTML structure, and not actually the page’s main content. Are there parts of the text you see that you think should have been filtered out by the extractor? Think about the quality of this text as training data: what might go wrong in training a model on text that looks like this? Conversely, what useful information can a model potentially extract from this page?

**Deliverable:** We should process the page content, filter out bad information and private information. For html structured data, it is best for us to extract the valid content from it.

(c) What makes a good training example is highly contextual. Describe an application domain for which this example might be useful to have in the training data, and one where it might not be.

**Deliverable:** A 1-2 sentence response.

(d) Let’s look at some more examples to get a better sense of what’s in the Common Crawl. Look through 25 more WET records. For each record, very briefly comment on the document’s language (if you can identify it), the domain name, what type of page it is, etc. How many examples does it take until you see what you’d deem a “high-quality” webpage?

**Deliverable:** Brief annotations of 25 documents with the document’s language, domain, type of page, and any other miscellaneous notes about the document. The number of examples it takes until you see a high-quality example.



### Problem (extract_text): 3 points

(a) Write a function that extracts text from a byte string containing raw HTML. Use resiliparse.extract.html2text.extract_plain_text to perform the extraction. This function needs a string, so you will need to first decode the byte string into a Unicode string. Be aware that the input byte string might not be encoded in UTF-8, so your function should be able to detect the encoding in case UTF-8 fails. Resiliparse also offers resiliparse.parse.encoding.detect_encoding(), which might be useful.

- [x] See function `extract_text_from_html_bytes` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_extract_text_from_html_bytes
```

(b) Run your text extraction function on a single WARC file. Compare its output to the extracted text in the corresponding WET file. What differences and/or similarities do you notice? Which extraction seems better?

**Deliverable:** The meaningless characters that I extract myself will have many more space breaks





### Problem (language_identification): 6 points

(a) Write a function that will take a Unicode string and identify the main language that is present in this string. Your function should return a pair, containing an identifier of the language and a score between 0 and 1 representing its confidence in that prediction.

```shell
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

- [x] See function `identify_language` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_identify_language
```

(b) The behavior of language models at inference time largely depends on the data they were trained on. As a result, issues in the data filtering pipeline can result in problems downstream. What issues do you think could arise from problems in the language identification procedure? In a higher-stakes scenario (such as when deploying a user-facing product), how would you go about mitigating these issues?

**Deliverable:** If we attempt to train a model that performs well in a specific language,  the wrong language will make the training even more confusing.  We can identify the training corpus with the highest confidence level by only training the language.

(c) Run your language identification system on text extracted from the WARC files (via your previously-implemented text extraction function). Manually identify the language in 20 random examples and compare your labels with the classifier predictions. Report any classifier errors. What fraction of documents are English? Based on your observations, what would be a suitable classifier confidence threshold to use in filtering?

**Deliverable:** A 2-5 sentence response.



### Problem (mask_pii): 3 points

1. Write a function to mask out emails. Your function will take a string as input, and replace all instances of email addresses with the string "|||EMAIL_ADDRESS|||". To detect email addresses, you can look up regular expressions that do this reliably.

	**Deliverable:** A function that replaces all email addresses in a given string with the string"|||EMAIL_ADDRESS|||", returning a pair containing both the new string and the number of instances that were masked. 

- [x] See function `mask_email` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_mask_emails
```



2. Write a function to mask out phone numbers. Your function will take a string as input, and replace all instances of phone numbers with the string "|||PHONE_NUMBER|||". Doing this reliably can be extremely challenging, as phone numbers might be written in an extremely diverse set of formats, but you should try to capture at least the most common phone number formats used in the United States, and be robust to minor syntactic deviations.

	**Deliverable:** A function that replaces phone numbers in a given string with the string "|||PHONE_NUMBER|||", returning a pair containing both the new string and the number of instances that were masked.

- [x] See function `mask_phone_numbers` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_mask_phones
```



3. Write a function to mask out IP addresses. For this problem, it is enough to focus on IPv4 addresses (4 numbers up to 255 separated by points). Your function will take a string as input, and replace all instances of IP addresses with the string "|||IP_ADDRESS|||". 

	**Deliverable:** A function that replaces IPv4 addresses in a given string with the string "|||IP_ADDRESS|||", returning a pair containing both the new string and the number of instances that were masked.

- [x] See function `mask_ip_address` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_mask_ips
```



4. What problems do you think might arise downstream in a language model when these filters are naïvely applied on the training set? How might you mitigate these issues?

	**Deliverable:** A 2-5 sentence response.

5. Run your PII masking functions on text extracted from the WARC files (via your previouslyimplemented text extraction function). Look through 20 random examples where a replacement was made; give some examples of false positives and false negatives.

	**Deliverable:** A 2-5 sentence response.



### Problem (harmful_content): 6 points

```shell
wget dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_nsfw_final.bin
wget dolma-artifacts.org/fasttext_models/jigsaw_fasttext_bigrams_20230515/jigsaw_fasttext_bigrams_hatespeech_final.bin
```

1. Write a function to detect NSFW content. 

	**Deliverable:** A function that labels a given string as containing NSFW content or not, returning a pair containing both the label and a confidence score.  Note that this test is just a sanity check, taken from the Jigsaw dataset, but by no means asserts that your classifier is accurate, which you should validate.

- [x] See function `classify_nsfw` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_classify_nsfw
```



2. Write a function to detect toxic speech.

	**Deliverable**: A function that labels a given string as consisting of toxic speech or not, returning a pair containing both the label and a confidence score.. Again, this test is just a sanity check, also taken from Jigsaw.

- [x] See function `classify_toxic_speech` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_classify_toxic_speech
```



3. What problems do you think might arise downstream in a language model when these filters are applied to create the training set? How might you mitigate these issues?

	**Deliverable:** A 2-5 sentence response.

4. Run your harmful content filters on text extracted from the WARC files (via your previouslyimplemented text extraction function). Look through 20 random examples and compare the classifier predictions to your own judgments. Report any classifier errors. What fraction of documents are harmful? Based on your observations, what would be suitable classifier confidence threshold(s) to use in filtering?

	**Deliverable:** A 2-5 sentence response.



### Problem (gopher_quality_filters): 3 points

(a) Implement (at least) the subset of the Gopher quality filters as described above. For tokenizing text into words, you might find the NLTK package useful (specifically nltk.word_tokenize), though you’re not required to use it.

- [x] See function `gopher_quality_filter` in `cs336_data/utils.py`

```python
# test shell
pytest -k test_gopher
```



(b) Run your rule-based quality filter on text extracted from the WARC files (via your previouslyimplemented text extraction function). Look through 20 random examples and compare the filter predictions to your own judgment. Comment on any cases where the quality filters differ from your judgments.

**Deliverable:** A 2-5 sentence response.



### Problem (quality_classifier): 15 points

```shell
wget https://nlp.stanford.edu/data/nfliu/cs336-spring-2024/assignment4/enwiki-20240420-extracted_urls.txt.gz
```

(a) Train a quality classifier that, given text, returns a numeric quality score.

**Deliverable:** A quality classifier for use in the next subproblem.

(b) Write a function that labels a page as high or low-quality, and provides a confidence score in the label.

**Deliverable:** A function taking a string as its only argument, and returning a pair with a label (high-quality or not) and a confidence score.

- [x] See code in `cs336_data/quality_classifier.py`

```python
# test shell
pytest -k test_classify_quality
```



## 3 Deduplication

### Problem (exact_deduplication): 3 points

Write a function that takes a list of paths to input files and performs exact line deduplication on them. It should first count the frequency of each line in the corpus, using a hash to reduce memory, and then rewrite each file by only keeping its unique lines.

**Deliverable:** A function that performs exact line deduplication. Your function should take two arguments: (a) a list of paths to its input files, and (b) an output directory. It should rewrite each input file to the output directory with the same name, but deduplicate the content by removing lines that occur more than once in the set of input files. For example, if the input paths are a/1.txt and a/2.txt, and the output directory is b/, your function should write the files b/1.txt and b/2.txt.

See function `exact_line_deduplication` in `cs336_data/utils.py`

```shell
# test shell
pytest -k test_exact_line_deduplication
```



### Problem (minhash_deduplication): 8 points

Write a function that takes a list of paths to input files and performs fuzzy document deduplication with minhash and LSH. In particular, your function should compute minhash signatures for each document in the provided list of paths, use LSH with the provided number of bands to identify candidate duplicates, and then compute the true ngram Jaccard similarity between candidate duplicates and remove those that exceed a given threshold. To improve recall (following Penedo et al., 2023), normalize the text before computing minhash signatures and/or comparing Jaccard similarity by lowercasing, removing punctuation, normalizing whitespaces, and removing accents, and applying NFD unicode normalization.

**Deliverable:** A function that performs fuzzy document deduplication. Your function should take at least the following arguments: (a) a list of paths to its input files, (b) the number of hashes to use for computing minhash signatures, (c) the number of bands to use for LSH, (d) the n-gram length (in words) for computing minhash signatures, and (e) an output directory. You may assume that the number of hashes to use for computing minhash signatures is evenly divisible by the number of bands to use for LSH.

Your function should rewrite each input file to the output directory with the same name, but only writing documents that are either (a) not candidate duplicates and/or (b) are randomly selected to be retained from the clustered buckets. For example, if the input paths are a/1.txt and a/2.txt, and the output directory is b/, your function should write the files b/1.txt and b/2.txt.

See function `minhash_deduplication` in `cs336_data/utils.py`

```shell
# test shell
pytest -k test_minhash_deduplication
```



## 4 Leaderboard: filter data for language modeling

pass