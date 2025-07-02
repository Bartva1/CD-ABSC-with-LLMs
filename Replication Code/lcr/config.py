from transformers import BertTokenizer, BertModel
import torch

#import torch_xla
#import torch_xla.core.xla_model as xm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#device = xm.xla_device()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')#.to(device)
model_bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True).to(device)
for param in model_bert.parameters():
    param.requires_grad = False

MAX_LENGTH = 136
MAX_LENGTH_LEFT = 102
MAX_LENGTH_RIGHT = 133
MAX_LENGTH_TARGET = 25
temperature = 0.001

descriptor_dimension = 200
batch_size = 1
num_epochs = 20

hops = 3
hidden_lstm = 300
hidden_dim = 768
output_size = 2
hidden_size = 128
num_domains = 3
num_polarities = 3

dropout_prob1 = 0.7
dropout_prob2 = 0.7

eps = 0.0050
alpha = 1.0
domains = ['book_2019','laptop_2014','restaurant_2014']
