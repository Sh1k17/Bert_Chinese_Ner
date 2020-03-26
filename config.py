from sklearn.metrics import f1_score, classification_report
import numpy as np
from pytorch_pretrained import BertModel, BertTokenizer
import torch
from utils import built_train_dataset,built_dev_dataset
class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'Bert_Bilstm_crf'

        # self.train_data_path = './datas/train/source.txt'  # 文本训练集
        # self.train_label_path = './datas/train/target.txt'  # 标签训练集
        self.train_data_path = './datas/train/train_source.txt'   # 文本验证集
        self.train_label_path = './datas/train/train_target.txt' # 标签验证集
        self.dev_data_path = './datas/dev/source.txt'  # 文本验证集
        self.dev_label_path = './datas/dev/target.txt'  # 标签验证集
        self.save_path = './Result/Save_path/' + self.model_name + '.ckpt'  # 模型训练结果
        self.bert_path = './bert_pretrain'

        self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')
        self.tokenizer = BertTokenizer.from_pretrained('./bert_pretrain')

        self.vocab_class ={'B-LAW': 0, 'B-ROLE': 1, 'B-TIME': 2, 'I-LOC': 3, 'I-LAW': 4, 'B-PER': 5, 'I-PER': 6,
                           'B-ORG': 7,
                           'I-ROLE': 8, 'I-CRIME': 9, 'B-CRIME': 10, 'I-ORG': 11, 'B-LOC': 12, 'I-TIME': 13, 'O': 14}# 词性类别名单
        self.tagset_size = len(self.vocab_class)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.num_epochs = 6  # epoch数
        self.batch_size = 2
        self.pad_size = 10  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5 # 学习率
        self.learning_rate_decay = 5e-6  # 学习率衰减
        self.hidden_size = 100
        self.embedding_dim = 768
        self.num_layers = 1
        self.dropout = 0.5