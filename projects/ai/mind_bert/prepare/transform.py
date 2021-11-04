#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   transform.py
#        \author   chenghuige  
#          \date   2020-04-24 17:08:23.881317
#   \Description  
# ==============================================================================

  
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
import os

# from https://www.kaggle.com/miklgr500/jigsaw-tpu-bert-two-stage-training

class TextTransformation:
    def __call__(self, text: str, lang: str = None) -> tuple:
        raise NotImplementedError('Abstarct')   
        
class LowerCaseTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        return text.lower(), lang
    
    
class URLTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        for url in self.find_urls(text):
            if url in text:
                text.replace(url, ' external link ')
        return text.lower(), lang
    
    @staticmethod
    def find_urls(string): 
        # https://www.geeksforgeeks.org/python-check-url-string/
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string) 
        return urls 
    
class PunctuationTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        for p in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’' +"/-'" + "&" + "¡¿":
            if '’' in text:
                text = text.replace('’', f' \' ')
                
            if '’' in text:
                text = text.replace('’', f' \' ')
              
            if '—' in text:
                text = text.replace('—', f' - ')
                
            if '−' in text:
                text = text.replace('−', f' - ')   
                
            if '–' in text:
                text = text.replace('–', f' - ')   
              
            if '“' in text:
                text = text.replace('“', f' " ')   
                
            if '«' in text:
                text = text.replace('«', f' " ')   
                
            if '»' in text:
                text = text.replace('»', f' " ')   
            
            if '”' in text:
                text = text.replace('”', f' " ') 
                
            if '`' in text:
                text = text.replace('`', f' \' ')              

            text = text.replace(p, f' {p} ')
                
        return text.strip(), lang
    
    
class NumericTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        for i in range(10):
            text = text.replace(str(i), f' {str(i)} ')
        return text, lang
    
class WikiTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        text = text.replace('wikiproject', ' wiki project ')
        for i in [' vikipedi ', ' wiki ', ' википедии ', " вики ", ' википедия ', ' viki ', ' wikipedien ', ' википедию ']:
            text = text.replace(i, ' wikipedia ')
        return text, lang
    
    
class MessageTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        text = text.replace('wikiproject', ' wiki project ')
        for i in [' msg ', ' msj ', ' mesaj ']:
            text = text.replace(i, ' message ')
        return text, lang
    
    
class PixelTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        for i in [' px ']:
            text = text.replace(i, ' pixel ')
        return text, lang
    
    
class SaleBotTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        text = text.replace('salebot', ' sale bot ')
        return text, lang
    
    
class RuTransformation(TextTransformation):
    def __call__(self, text: str, lang: str = None) -> tuple:
        if lang is not None and lang == 'ru' and 'http' not in text and 'jpg' not in text and 'wikipedia' not in text:
            text = text.replace('t', 'т')
            text = text.replace('h', 'н')
            text = text.replace('b', 'в')
            text = text.replace('c', 'c')
            text = text.replace('k', 'к')
            text = text.replace('e', 'е')
            text = text.replace('a', 'а')
        return text, lang
    
class CombineTransformation(TextTransformation):
    def __init__(self, transformations: list, return_lang: bool = False):
        self._transformations = transformations
        self._return_lang = return_lang
        
    def __call__(self, text: str, lang: str = None) -> tuple:
        for transformation in self._transformations:
            text, lang = transformation(text, lang)
        if self._return_lang:
            return text, lang
        return text
    
    def append(self, transformation: TextTransformation):
        self._transformations.append(transformation)

 
transformer = CombineTransformation(
    [
        LowerCaseTransformation(),
        PunctuationTransformation(),
        NumericTransformation(),
        PixelTransformation(),
        MessageTransformation(),
        WikiTransformation(),
        SaleBotTransformation()
    ]
)

 
