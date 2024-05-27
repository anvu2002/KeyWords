"""
https://github.com/google/sentencepiece#installation
Legacy V/s New Behavior: https://github.com/huggingface/transformers/pull/24565
"""

from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.tokenize import sent_tokenize, word_tokenize


model = T5ForConditionalGeneration.from_pretrained("Voicelab/vlt5-base-keywords")
tokenizer = T5Tokenizer.from_pretrained("Voicelab/vlt5-base-keywords")

def filter_keywords(file : str) -> list:
    task_prefix = "Keywords: "


    fp = open(file, encoding='UTF-8')
    raw_text = fp.read()
    sentences = sent_tokenize(raw_text)

    inputs_2 = sentences
    predicted = []

    for sample in inputs_2:
        input_sequences = [task_prefix + sample]
        input_ids = tokenizer(
            input_sequences, return_tensors="pt", truncation=True
        ).input_ids
        
        output = model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
        predicted.append(tokenizer.decode(output[0], skip_special_tokens=True))
        # print(sample, "\n --->", predicted)
    return sentences,predicted

sen,res = filter_keywords("Human_Nature.txt")
breakpoint()