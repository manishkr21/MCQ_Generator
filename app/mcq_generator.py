import string
import requests
import argparse
import pprint
from urllib.parse import quote as url_quote
import torch
from torch.nn.functional import softmax
import nltk
from nltk.corpus import wordnet as wn
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from allennlp.predictors.predictor import Predictor
from summa.summarizer import summarize
from sense2vec import Sense2Vec
import pke

class BertWSD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.ranking_linear = torch.nn.Linear(config.hidden_size, 1)
        self.init_weights()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('loading BERT-WSD...')
model_path = '../models/bert_base-augmented-batch_size=128-lr=2e-5-max_gloss=6'
model = BertWSD.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenizer.added_tokens_encoder['[TGT]'] = 100
model.to(device)
model.eval()

print('loading question model...')
question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base')

print('loading conref predictor...')
coref_predictor = Predictor.from_path('../models/coref-spanbert')

print('loading Sense2Vec...')
s2v = Sense2Vec().from_disk('../models/s2v_old')

print('All models loaded.')

def get_wordnet_choices(syn, answer: str) -> list[str]:
    choices = []
    hypernyms = syn.hypernyms()
    if not hypernyms:
        return []
    for hyponym in hypernyms[0].hyponyms():
        choice = hyponym.lemmas()[0].name()
        if choice == answer:
            continue
        choice = choice.replace('_', ' ')
        choice = string.capwords(choice)
        if choice and choice not in choices:
            choices.append(choice)
    return choices


def get_conceptnet_choices(answer: str) -> list[str]:
    parent_request_url = f'http://api.conceptnet.io/query?node=/c/en/{answer}/n&rel=/r/PartOf&start=/c/en/{answer}&limit=5'
    response = requests.get(parent_request_url).json()
    choices = []
    for edge in response['edges']:
        end_term = edge['end']['term']
        children_request_url = f'http://api.conceptnet.io/query?node={end_term}&rel=/r/PartOf&end={end_term}&limit=8'
        child_response = requests.get(children_request_url).json()
        for child_edge in child_response['edges']:
            choice = child_edge['start']['label']
            if choice not in choices and choice.replace(' ', '_').lower() not in answer:
                choices.append(choice)
    return choices


def get_s2v_choices(answer: str) -> list[str]:
    global s2v
    choices = []
    sense = s2v.get_best_sense(answer)
    if sense:
        most_similar = s2v.most_similar(sense)
    else:
        return []
    for word in most_similar:
        word = word[0].split('|')[0].replace('_', ' ')
        if word not in choices and word.lower() not in answer.replace('_', ' '):
            choices.append(word)
    
    return choices


def generate_wrong_choices(syn, answer: str) -> list[str]:
    choices = []
    answer = answer.lower().strip().replace(' ', '_')
    if syn:
        wordnet_choices = get_wordnet_choices(syn, answer)
        if wordnet_choices:
            if len(wordnet_choices) >= 4:
                return wordnet_choices
            else:
                choices.extend(wordnet_choices)
        else:
            pass # TODO: Log
    
    try:
        conceptnet_choices = get_conceptnet_choices(answer)
    except:
        conceptnet_choices = None
        
    if conceptnet_choices:
        choices.extend(conceptnet_choices)
    else:
        pass # TODO: log

    s2v_choices = get_s2v_choices(answer)
    if s2v_choices:
        choices.extend(s2v_choices)
    else:
        pass # TODO: log

    return list(set(choices))


def get_features(sentence, definition, max_seq_len: int, tokenizer):
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token = 0

    cls_token_type_id = 1
    pad_token_type_id = 0

    sentence_tokens = tokenizer.tokenize(sentence)

    features = []
    for sequence in definition:
        definition_tokens = tokenizer.tokenize(sequence)
        sentence_tokens, definition_tokens = truncate_sequences(sentence_tokens, definition_tokens, max_seq_len - 3)

        tokens = [cls_token]
        token_type_ids = [cls_token_type_id]
        
        tokens += sentence_tokens + [sep_token]
        token_type_ids += [0] * (len(sentence_tokens) + 1)

        tokens += definition_tokens + [sep_token]
        token_type_ids += [1] * (len(definition_tokens) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        pad_len = max_seq_len - len(input_ids)
        input_ids += ([pad_token] * pad_len)
        input_mask += ([0] * pad_len)
        token_type_ids += ([pad_token_type_id] * pad_len)

        features.append({
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': token_type_ids,
        })
    
    return features


def truncate_sequences(a: list, b:list, max_len: int):
    len_a = len(a)
    len_b = len(b)
    tot_len = len_a + len_b
    if tot_len <= max_len:
        return a, b

    half_len = max_len // 2
    if len_a >= half_len and len_b >= half_len:
        return b[:half_len], b[:half_len]
    else:
        if len_a > len_b:
            return a[:-(tot_len - max_len)], b
        else:
            return a, b[:-(tot_len - max_len)]


MAX_SEQ_LEN = 128

def get_sense(sentence: str, ambiguous_word: str):
    senses = []
    definitions = []
    for synset in wn.synsets(ambiguous_word, 'n'):
        senses.append(synset)
        definitions.append(synset.definition())
    
    if not senses:
        return None

    features = get_features(sentence, definitions, MAX_SEQ_LEN, tokenizer)

    with torch.no_grad():
        logits = torch.zeros(len(definitions), dtype=torch.double).to(device)
        for i, bert_input in enumerate(features):
            bert_input = {k: torch.tensor(v, dtype=torch.long).unsqueeze(0).to(device) for k, v in bert_input.items()}
            logits[i] = model.ranking_linear(model.bert(**bert_input)[1])
        
        scores = softmax(logits, dim=0)

        predictions = sorted(zip(senses, definitions, scores), key=lambda x: x[2], reverse=True)
    
    sense = predictions[0][0]
    return sense


def get_question(sentence: str, keyword: str) -> str:
    global question_model, question_tokenizer
    text = f'context: {sentence} answer: {keyword}'
    max_len = 256
    encoding = question_tokenizer.encode_plus(text, max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')

    outputs = question_model.generate(input_ids=encoding['input_ids'],
                                      attention_mask=encoding['attention_mask'],
                                      early_stopping=True,
                                      num_beams=5,
                                      num_return_sequences=1,
                                      no_repeat_ngram_size=2,
                                      max_length=200)
    
    question = question_tokenizer.decode(outputs[0], skip_special_tokens=True)
    question = question.removeprefix('question: ')
    return question


def place_tags(sentence: str, keyword: str, tag: str) -> str:
    index = sentence.lower().find(keyword)
    if index == -1:
        return sentence

    kw_len = len(keyword)
    return (sentence[:index - 1 if index != 0 else index] + \
     tag + sentence[index: index + kw_len] + tag + \
         sentence[index + kw_len + 1:]).strip()


def generate_mcq_from_sentence(sentence: str, keyword: str) -> dict:
    sentence_bert = place_tags(sentence, keyword, ' [TGT] ')
    sense = get_sense(sentence_bert, keyword)
    choices = generate_wrong_choices(sense, keyword)
    question = get_question(sentence, keyword)
    if keyword[0].isalpha() and keyword[0].islower():
        keyword = list(keyword)
        keyword[0] = keyword[0].upper()
        keyword = ''.join(keyword)
    return {'question': question, 'answer': keyword, 'choices': choices}


def extract_sentences(text: str) -> list[str]:
    global coref_predictor
    text = coref_predictor.coref_resolved(text)
    sentences = summarize(text, ratio=1, split=True)
    if len(sentences) == 0:
        return [text]

    return sentences


def extract_keywords_from_sentences(sentences: list[str]) -> dict[str, list[str]]:
    pos = {'NOUN', 'PROPN'}

    keyword_extracter = pke.unsupervised.MultipartiteRank()
    sentence_keywords = {}
    for s in sentences:
        keyword_extracter.load_document(input=s, language='en')
        keyword_extracter.candidate_selection(pos=pos)
        try:
            keyword_extracter.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        except:
            sentence_keywords[s] = []
            continue
        keywords = keyword_extracter.get_n_best(n=5)
        result = []
        for keyword in keywords:
            result.append(keyword[0])
        sentence_keywords[s] = result
    return sentence_keywords


def generate_mcqs(text: str, max_choices: int=4) -> list[dict]:
    sentences = extract_sentences(text)
    sentence_keywords = extract_keywords_from_sentences(sentences)
    mcqs = []
    answers = set()
    for sentence, keywords in sentence_keywords.items():
        if not keywords:
            continue
        mcq = generate_mcq_from_sentence(sentence, keywords[0])
        if max_choices !=  -1:
            if len(mcq['choices']) > max_choices - 1:
                mcq['additional_choices'] = mcq['choices'][max_choices - 1: max_choices + 5]
            mcq['choices'] = mcq['choices'][:max_choices - 1]

        mcqs.append(mcq)
        if mcq['answer'] in answers and len(keywords) > 1:
            mcq = generate_mcq_from_sentence(sentence, keywords[1])
            if max_choices !=  -1:
                mcq['choices'] = mcq['choices'][:max_choices - 1]
            mcqs.append(mcq) 
    return mcqs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', action='store', type=str, required=True)
    parser.add_argument('--max-choices', action='store', type=int)
    args = parser.parse_args()
    if args.max_choices:
        max_choices = args.max_choices
    else:
        max_choices = 4
    pprint.pprint(generate_mcqs(args.text, max_choices=max_choices))


