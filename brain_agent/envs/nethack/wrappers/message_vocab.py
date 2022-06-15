
import gym
import re
from nle.env.tasks import NetHackChallenge, NetHackScore
from nle import nethack
from brain_agent.utils.utils import log
from typing import Union
import nle
import json
import numpy as np
import torch
import string
import nltk

import os
from torchtext.data.utils import get_tokenizer

ASSETS_PATH = os.path.join(os.path.dirname(__file__), '..', 'anssi', 'assets')




def rfind_any(s, sub_list):
    for sub in sub_list:
        p = s.rfind(sub)
        if p >= 0:
            return p
    return -1


def _text_pipeline(text, embedding_dim):
    n = embedding_dim  # text.shape[0]
    text = text.tobytes().strip(b'\0').decode(errors='ignore')
    text = re.sub('[\[\(].*[\]\)]?', '', text)
    text = re.sub('\d+', '', text)
    text = re.sub('\".*\"?', '', text)

    p = rfind_any(text, list(string.punctuation))
    if p >= 0:
        text = text[:p + 1]

    _tokenizer = get_tokenizer('basic_english')
    tokens = _tokenizer(text)[:n]
    embed = np.array(_torchtext_vocab(tokens))
    embed_pad = np.pad(embed, (0, n - len(embed)), mode='constant', constant_values=0)
    return torch.tensor(embed_pad)


_torchtext_vocab = torch.load(os.path.join(ASSETS_PATH, 'torchtext_vocab.pth'))
log.info(f'vocab size: {len(_torchtext_vocab)}')


p1 = re.compile('\?\s+\[[\d\w\W]+\]')
p2 = re.compile('[\'\"]\s+[\'\"]')
p3 = re.compile('\s{2,}')
p4 = re.compile('\d[\:\d]*')
p5 = re.compile('\|[\s]*')
not_allowed_messages = [
    '#',
    '^',
    'call a',
    'what type of scroll do you want to write?',
    'adjust letter',
    'you read:',
    'the engraving now reads:',
    'you feel the words:',
    'what do you want to name'
]


def is_allowed(text):
    for sent_pattern in not_allowed_messages:
        if text.startswith(sent_pattern):
            return False
    return True


def preprocess(text):
    text = text.lower().replace('--more--', '')
    search_item = p1.search(text)

    if not is_allowed(text):
        return ''

    if search_item is not None:
        text = text[:search_item.start(0) + 1]
    text = re.sub(p2, '', text)
    text = re.sub(p3, ' ', text)
    text = re.sub(p4, ' NUM ', text)

    read_idx = text.rfind('you read:')
    if read_idx > -1:
        text = text[:read_idx]

    return text.strip()


with open(os.path.join(ASSETS_PATH, 'vocab.json')) as reader:
    _nltk_vocab = json.load(reader)


def _nltk_word_ids(message, embedding_dim):
    message = message.tobytes().strip(b'\0').decode(errors='ignore')

    message = preprocess(message)
    tokens = nltk.word_tokenize(message)
    ids = np.array([_nltk_vocab.get(tok, 0) for tok in tokens])
    embed_pad = np.pad(ids, (0, max(0, embedding_dim - len(ids))), mode='constant', constant_values=0)
    return torch.tensor(embed_pad)


class MessageVocabEmbeddingWrapper(gym.Wrapper):
    def __init__(self, env, tokenizer, embedding_dim=None, **kwargs):
        super().__init__(env)
        self.tokenizer = tokenizer
        self._vocab = {
            'torchtext': _torchtext_vocab,
            'nltk': _nltk_vocab,
        }[tokenizer]
        self._pipeline = {
            'torchtext': _text_pipeline,
            'nltk': _nltk_word_ids,
        }[tokenizer]
        self._embedding_dim = 32 if tokenizer == 'torchtext' else 300

        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces['message_embedding'] = gym.spaces.Box(
            low=0,
            high=len(self._vocab),
            shape=(self._embedding_dim,),
            dtype=np.int64
        )
        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _create_message_embedding(self, observation):
        message = observation['message']
        observation['message_embedding'] = self._pipeline(message, self._embedding_dim)

    def step(self, action: int):
        observation, reward, done, info = self.env.step(action)
        self._create_message_embedding(observation)
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self._create_message_embedding(observation)
        return observation

    ## for submission below here
    def on_reset_submission(self, env, obs, is_holding_aicrowd=False):
        self.is_holding_aicrowd = is_holding_aicrowd
        self.env_aicrowd = env
        #self.step = lambda action: self.step_submission(action, self.env_aicrowd)

        self._create_message_embedding(obs)

        return obs