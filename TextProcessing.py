#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:55:49 2019

@author: niraj
"""
from os import path
root_dir = path.dirname(path.dirname(__file__))
import sys
if root_dir not in sys.path:
    sys.path.append(root_dir)
import re
from os import path
from tqdm import tqdm
import pandas as pd
from ArgsManager import ArgsManager
from abc import ABC,abstractmethod
import spacy
import processing_metadata as metadata

def replace_all(text, value, replacement):
    while value in text:
        text = text.replace(value, replacement)
    return text

class TextProcessingInterface(ABC):
    
    @abstractmethod
    def parse_texts(self, texts:[]):
        pass
    
    @abstractmethod
    def parse_text(self, text:str):
        pass

class SimpleProcessing(TextProcessingInterface):
    
    def __init__(self, args:ArgsManager, model=None):
        self.args = args       
        self.apply_lemma = args['nlp_apply_lemma']
        self.cleanse_string = args['nlp_cleanse_text']
        self.nlp_remove_stop_words = args['nlp_remove_stop_words']
        self.remove_verb_contractions = args['nlp_remove_verb_contractions']
        self.replace_extended_chars = args['nlp_replace_extended_chars']
        self.model = None
        print('INFO::', 'Pipeline Lemmatization is set to :',self.apply_lemma)
        print('INFO::', 'Pipeline Stopwords removal is set to :',self.nlp_remove_stop_words)
        print('INFO::', 'Pipeline Contraction removal is set to :',self.remove_verb_contractions)
        print('INFO::', 'Pipeline Latin extended char removal is set to :',self.replace_extended_chars)
        print('INFO::', 'Pipeline Cleansing is set to :', self.cleanse_string)
        if self.apply_lemma:
            if model is not None:
                self.model = model
            else:
                print('loading language model:',args['nlp_model_name'])
                self.model = spacy.load(args['nlp_model_name'])
                print('model loaded!')
        self.stopwords = metadata.stopwords
        self.latin_extended_char_mapping = metadata.latin_extended_char_mapping

    def decontract(self, phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"ain\'t", "is not", phrase)
        phrase = re.sub(r"shan\'t", "will not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase

    def replace_extended_characters(self, text):
        for latin_char in self.latin_extended_char_mapping:
            if latin_char in text:
                text = text.replace( latin_char, self.latin_extended_char_mapping[latin_char])
        return text
        
    def parse_text(self, text:str):
        if self.apply_lemma:
            result = []
            for token in self.model(text):
                if self.nlp_remove_stop_words and token.text in self.stopwords:
                    continue
                if 'PRP' not in token.tag_ and len(token.lemma_)>0:
                    result.append(token.lemma_)
                else:
                    result.append(token.text)
            text = ' '.join(result)
        if self.nlp_remove_stop_words and (self.apply_lemma):
            text = ' '.join([tk for tk in text.split() if tk.lower() not in self.stopwords])
        if self.remove_verb_contractions:
            text = self.decontract( text)
        if self.replace_extended_chars:
            text = self.replace_extended_characters( text)        
        if self.cleanse_string:
            text = re.sub('[^a-zA-Z?\s]',' ? ',text)
        text = replace_all(text, '  ', ' ').lstrip().rstrip().lower()
        text = replace_all(text,'? ?','?')
        return text

    # def parse_series(self, df:pd.Series):
    #     for i,text in tqdm(df.iteritems(), total = df.count()):
    #         df.iloc[i] = self.parse_text(text)
    #     return df
    
    def parse_texts(self, texts:[]):
        for i, text in tqdm(enumerate(texts), total= len(texts)):
            texts[i] = self.parse_text(text)
        return texts

class ComplexProcessing(TextProcessing):

    
    def __init__(self, args:ArgsManager, model=None, remove_chars:[] = ['?','_', '"','``','-',"''"]):
        if not model:
            print('loading language model:',args['nlp_model_name'],'..')
            self.model = spacy.load(args['nlp_model_name'])
        else:
            print('using existing model')
            self.model = model
        print('loading entity dictionary')
        self.entity_mapping = pm.load_data(path.join('pickles','entity_mapping.pkl'))
        self.entity_keys = list(self.entity_mapping.keys())
        self.entity_values = list(self.entity_mapping.values())
        print('nlp metadata loaded!')
        self.apply_lemma = args['nlp_apply_lemma']
        self.remove_stop_words = args['nlp_remove_stop_words']
        self.remove_chars = remove_chars
        self.remove_entities = args['nlp_simplify_entities']
        #code to take care of incorrect negations
        

    def decontract(self, phrase):
        # specific
        phrase = re.sub(r"won\'t", "will not", phrase)
        phrase = re.sub(r"can\'t", "can not", phrase)
        phrase = re.sub(r"ain\'t", "is not", phrase)
        phrase = re.sub(r"shan\'t", "will not", phrase)

        # general
        phrase = re.sub(r"n\'t", " not", phrase)
        phrase = re.sub(r"\'re", " are", phrase)
        phrase = re.sub(r"\'s", " is", phrase)
        phrase = re.sub(r"\'d", " would", phrase)
        phrase = re.sub(r"\'ll", " will", phrase)
        phrase = re.sub(r"\'t", " not", phrase)
        phrase = re.sub(r"\'ve", " have", phrase)
        phrase = re.sub(r"\'m", " am", phrase)
        return phrase
        
    def __postprocessing__(self, text):
        if self.remove_chars:
            for ch in self.remove_chars:
                text = text.replace(ch,' ')
        text = text.replace('  ',' ')
        return text
    
    def __preprocessing__(self, text):
        for latin_char in self.latin_extended_char_mapping:
            if latin_char in text:
                text = text.replace( latin_char, self.latin_extended_char_mapping[latin_char])
        text = self.decontract( text)
        return text
        
    def parse_with_entities(self,text:str):
        text = self.__preprocessing__(text)
        doc = self.model(text)
        result = []
        for token in doc:
            if self.remove_stop_words and token.is_stop:
                continue
            if self.apply_lemma and 'PRP' not in token.tag_:
                tk_text = token.lemma_.lower()
            else:
                tk_text = token.text.lstrip().rstrip().lower()
            result.append( tk_text)
        text = ' '.join(result)
        return self.__postprocessing__(text)
    
    def parse_without_entities(self, text:str):
        text = self.__preprocessing__(text)
        doc = self.model(text)
        result = []
        last_entity =None
        last_entity_word = ''
        for tk in doc:
            if self.remove_stop_words and tk.is_stop:
                continue
            tk_text = tk.text.lower().lstrip().rstrip()
            #check for entities
            if tk.ent_iob_=='O':
                if self.apply_lemma and 'PRP' not in tk.tag_ : #spacy does this weird conversion for pro-nouns
                    result.append(tk.lemma_.lower())
                else:
                    result.append(tk_text)
                last_entity =None
                last_entity_word = ''
            elif last_entity == tk.ent_type_:
                last_entity_word +=  tk.text
                continue
            else:
                last_entity_word +=  tk.text
                result.append(self.entity_mapping[tk.ent_type_])
                last_entity = tk.ent_type_
        text = ' '.join(result)
        return self.__postprocessing__(text)
    
    def parse_texts(self, texts:[]):
        result = []
        for text in texts:
            if self.remove_entities:
                result.append(self.parse_without_entities(text))
            else:
                result.append(self.parse_with_entities(text))
        return result
    
    def parse_series(self, df:pd.Series):
        for i,text in df.iteritems():
            if self.remove_entities:
                df.iloc[i] = self.parse_without_entities(text)
            else:
                df.iloc[i] = self.parse_with_entities(text)
        return df
                
if __name__ == '__main__':

    args = ArgsManager('args.json')
    sp = SimpleProcessing(args)
    print(sp.parse_texts(['please don\'t change this ? i can\'t. This is the best']))
    print(sp.parse_text('aint doin it pun   kass 123 for rs 123.2 but   he doesnt doesnt isnt arent 123d13f1s3d  . '))
    
    
    
    #df.examples = nlPipeline.parse_series( df.examples)
