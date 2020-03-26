from pytorch_pretrained import BertModel,BertTokenizer
# -*- coding: utf-8 -*
import torch
from torch import nn
import random
from config import Config
random.seed(1)
START_TAG = "CLS"
STOP_TAG = "SEP"

Config = Config()
class BiLSTMCRF(nn.Module):
    def __init__(self):
        super(BiLSTMCRF, self).__init__()
        self.tag2id = {'B-LAW': 0, 'B-ROLE': 1, 'B-TIME': 2, 'I-LOC': 3, 'I-LAW': 4, 'B-PER': 5, 'I-PER': 6,
                           'B-ORG': 7,
                           'I-ROLE': 8, 'I-CRIME': 9, 'B-CRIME': 10, 'I-ORG': 11, 'B-LOC': 12, 'I-TIME': 13, 'O': 14, START_TAG: 15, STOP_TAG: 16}
        self.tag2id_size = len(self.tag2id)
        self.bert_path = './bert_pretrain'
        self.bert = BertModel.from_pretrained(self.bert_path)
        self.batch_size = Config.batch_size
        self.embedding_dim = 768
        self.hidden_dim = Config.hidden_size
        # 概率转移矩阵
        self.transitions = nn.Parameter(torch.randn(self.tag2id_size, self.tag2id_size))
        self.transitions.data[:, self.tag2id[START_TAG]] = -1000.
        self.transitions.data[self.tag2id[STOP_TAG], :] = -1000.
        self.layerNorm = nn.LayerNorm(self.embedding_dim)
        # embeddings
        # self.word_embeddings = nn.Embedding(self.word2id_size, self.embedding_dim)

        # batch_first=True：batch_size在第一维而非第二维
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers = 1, bidirectional = True,
                            batch_first = True,dropout=0.5)
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag2id_size)

    # 每帧对应的隐向量
    # input =[contex,mask]
    def get_lstm_features(self, batch_sentence,input_mask):
        context = batch_sentence
        mask = input_mask
        batch_size = context.size(0)
        seq_len = context.size(1)
        with torch.no_grad():
            embeddings, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
            # print(embeddings.shape)
        hidden = (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                  torch.randn(2, self.batch_size, self.hidden_dim // 2))

        lstm_out, _hidden = self.lstm(embeddings, hidden)
        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        # 降维到标签空间
        return self.hidden2tag(lstm_out)

    # 真实路径的分值（针对一个实例而非一个batch）
    def real_path_score(self, logits, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag2id[START_TAG]], dtype = torch.long), tags])
        # 累加每帧的转移和发射
        for i, logit in enumerate(logits):
            # len(tags) = len(logits) + 1
            transition_score = self.transitions[tags[i], tags[i + 1]]
            emission_score = logit[tags[i + 1]]
            score += transition_score + emission_score
        # 处理结尾
        score += self.transitions[tags[-1], self.tag2id[STOP_TAG]]
        return score

    # 数值稳定性
    def log_sum_exp(self, smat):
        vmax = smat.max(dim = 0, keepdim = True).values    # 每列的最大值
        return (smat - vmax).exp().sum(axis = 0, keepdim = True).log() + vmax

    # 概率归一化分母（针对一个实例而非一个batch）
    def total_score(self, logits):
        alpha = torch.full((1, self.tag2id_size), -1000.)
        alpha[0][self.tag2id[START_TAG]] = 0
        # 沿时间轴dp
        for logit in logits:
            alpha = self.log_sum_exp(alpha.T + logit.unsqueeze(0) + self.transitions)
        # STOP，发射分值0，转移分值为列向量（self.tag2id[STOP_TAG]外加上[]）
        return self.log_sum_exp(alpha.T + 0 + self.transitions[:, [self.tag2id[STOP_TAG]]]).flatten()

    # 负对数似然
    def neg_log_likelihood(self, batch_sentences, batch_tags, input_mask):

        batch_length = torch.sum(input_mask,axis = 1)
        batch_logits = self.get_lstm_features(batch_sentences,input_mask)
        real_path_score = torch.zeros(1)
        total_score = torch.zeros(1)
        # 一个batch求和
        for logits, tags, len in zip(batch_logits, batch_tags, batch_length):
            # mask

            logits = logits[:len]
            tags = tags[:len]
            # print(logits.shape,tags.shape)
            real_path_score += self.real_path_score(logits, tags)
            total_score += self.total_score(logits)

        return total_score - real_path_score

    # 维特比解码
    def viterbi_decode(self, logits):
        backtrace = []
        # 初始化
        alpha = torch.full((1, len(self.tag2id)), -1000.)
        alpha[0][self.tag2id[START_TAG]] = 0
        # 沿时间轴dp
        for frame in logits:
            smat = alpha.T + frame.unsqueeze(0) + self.transitions
            backtrace.append(smat.argmax(0))    # 当前时刻，每个状态的最优来源
            alpha = self.log_sum_exp(smat)
        smat = alpha.T + 0 + self.transitions[:, [self.tag2id[STOP_TAG]]]
        # 回溯路径
        best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        # 从[1:]开始，去掉START_TAG
        for bptrs_t in reversed(backtrace[1:]):
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        # 最优路径分值和最优路径
        return self.log_sum_exp(smat).item(), best_path[::-1]

    # 推断
    def forward(self, batch_sentences,input_mask):
        batch_sentences = torch.tensor(batch_sentences, dtype = torch.long)
        # batch_length = [each.size(-1) for each in batch_sentences]
        batch_length = torch.sum(torch.tensor(input_mask),axis = 1)
        # print(input_mask)
        batch_logits = self.get_lstm_features(batch_sentences,input_mask)

        batch_scores = []
        batch_paths = []
        # 计算一个batch
        for logits, len in zip(batch_logits, batch_length):
            logits = logits[:len]
            score, path = self.viterbi_decode(logits)
            batch_scores.append(score)
            batch_paths.append(path)
        return batch_scores, batch_paths