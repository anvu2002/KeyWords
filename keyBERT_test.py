﻿from keybert import KeyBERT

doc = """
         Supervised learning is the machine learning task of learning a function that
         maps an input to an output based on example input-output pairs. It infers a
         function from labeled training data consisting of a set of training examples.
         In supervised learning, each example is a pair consisting of an input object
         (typically a vector) and a desired output value (also called the supervisory signal).
         A supervised learning algorithm analyzes the training data and produces an inferred function,
         which can be used for mapping new examples. An optimal scenario will allow for the
         algorithm to correctly determine the class labels for unseen instances. This requires
         the learning algorithm to generalize from the training data to unseen situations in a
         'reasonable' way (see inductive bias). However, Machine Learning is indeed the begining a new
         digital era!
      """

def filter_keywords(file: str) ->str:
      fp = open(file, encoding='UTF-8')
      raw_text = fp.read()
      
      kw_model = KeyBERT()
      keywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None)
      
      return keywords

kw = filter_keywords("Human_Nature.txt")
breakpoint()