from torch.utils.data import DataLoader
import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from config import *
import random
import torch.nn.functional as F
import gc 
from masking_restraints import is_subsequence_stopwords2

def divide_samples(input_file, train_file, test_file, train_ratio=0.8):
    random.seed(123)
    # Read lines from input file
    with open(input_file, 'r', encoding='latin-1') as file:
        lines = file.readlines()

    # Group lines into samples (every 3 lines)
    samples = [lines[i:i+3] for i in range(0, len(lines), 3)]

    # Shuffle samples randomly
    random.shuffle(samples)

    # Split samples into training and testing sets
    train_samples = samples[:int(train_ratio * len(samples))]
    test_samples = samples[int(train_ratio * len(samples)):]

    # Write training samples to train_file
    with open(train_file, 'w', encoding='latin-1') as file:
        for sample in train_samples:
            file.writelines(sample)

    # Write testing samples to test_file
    with open(test_file, 'w', encoding='latin-1') as file:
        for sample in test_samples:
            file.writelines(sample)

def concatenate_four_files(file1_path, file2_path,file3_path, file4_path, output_file_path):
    with open(file1_path, 'r', encoding='latin-1') as file1:
        content1 = file1.read()

    with open(file2_path, 'r', encoding='latin-1') as file2:
        content2 = file2.read()

    with open(file3_path, 'r', encoding='latin-1') as file3:
        content3 = file3.read()
    
    with open(file4_path, 'r', encoding='latin-1') as file4:
        content4 = file4.read()
    with open(output_file_path, 'w', encoding='latin-1') as output_file:
        output_file.write(content1)
        output_file.write(content2)
        output_file.write(content3)
        output_file.write(content4)

def concatenate_two_files(file1_path, file2_path, output_file_path):
    with open(file1_path, 'r', encoding='latin-1') as file1:
        content1 = file1.read()

    with open(file2_path, 'r', encoding='latin-1') as file2:
        content2 = file2.read()

  
    with open(output_file_path, 'w', encoding='latin-1') as output_file:
        output_file.write(content1)
        output_file.write(content2)

def get_stats_from_file(path):
    """
    Method obtained from Trusca et al. (2020), no original docstring provided.

    :param path: file path
    :return:
    """
    polarity_vector = []
    with open(path, "r", encoding='latin-1') as fd:
        lines = fd.read().splitlines()
        size = len(lines) / 3
        
        for i in range(0, len(lines), 3):
            # Polarity.
            polarity_vector.append(lines[i + 2].strip().split()[0])
    polarity_vector = [int(x) for x in polarity_vector]        
    print(f'Positive sentiment: {polarity_vector.count(1)} {(100 * polarity_vector.count(1) / len(polarity_vector))}')
    print(f'Negative sentiment: {polarity_vector.count(-1)} {(100 * polarity_vector.count(-1) / len(polarity_vector))}')
    print(f'Neutral sentiment: {polarity_vector.count(0)} {(100 * polarity_vector.count(0) / len(polarity_vector))}')
    return size, polarity_vector

class CustomDataset(Dataset):
    def __init__(self, *tensors):
        self.tensors = tensors
        assert all(len(t) == len(tensors[0]) for t in tensors), "Length of tensors must be the same"

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        
        return tuple(tensor[idx] for tensor in self.tensors)

class CustomDataset2(Dataset):
    def __init__(self, *tensors, target_tensor_index, target_values):
        self.tensors = list(tensors)
        self.target_tensor_index = target_tensor_index
        self.target_values = target_values
        assert all(len(t) == len(tensors[0]) for t in tensors), "Length of tensors must be the same"

        # Filter indices where the target tensor has one of the target values
        self.filtered_indices = self._filter_indices()

        target_tensor = self.tensors[self.target_tensor_index]
        
        self.domain_tensor = []
        self.hot_domain = []
        for idx,value in enumerate(target_tensor):
            if value == target_values[0]:
                self.domain_tensor.append(0)
                self.hot_domain.append([1.0,0.0])
            elif value == target_values[1]:
                self.domain_tensor.append(1)
                self.hot_domain.append([0.0,1.0])
            else:
                self.domain_tensor.append(3)
                self.hot_domain.append([0.0,0.0])

        self.domain_tensor = torch.tensor(self.domain_tensor)
        self.hot_domain = torch.tensor(self.hot_domain)
        
    def _filter_indices(self):
        target_tensor = self.tensors[self.target_tensor_index]
        
        mask = torch.isin(target_tensor, torch.tensor(self.target_values))
        return mask.nonzero(as_tuple=True)[0]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        filtered_idx = self.filtered_indices[idx]

        self.tensors[self.target_tensor_index] = self.domain_tensor
        self.tensors[4] = self.hot_domain
        return tuple(tensor[filtered_idx] for tensor in self.tensors)
    
def get_tokenized(text):
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    #MAX_LENGTH = 87
    with torch.no_grad():
        tokenized_sentences = tokenizer(text,padding='max_length',max_length = MAX_LENGTH)
        tokens_tensor = torch.tensor([tokenized_sentences.get('input_ids')],device='cpu')
        segments_tensor = torch.tensor([tokenized_sentences.get('attention_mask')],device='cpu')
    del tokenized_sentences
   
    return tokens_tensor.squeeze(0),segments_tensor.squeeze(0)

def get_embeddings(tokens_tensor,segments_tensor):
    
    # Model evaluation
    model_bert.eval()
    with torch.no_grad():
       
        # Obtain hidden states
        tokens_tensor,segments_tensor = tokens_tensor.to(device),segments_tensor.to(device)
        outputs = model_bert(tokens_tensor, segments_tensor)
        
        hidden_states = torch.sum(torch.stack(outputs.hidden_states[-4:],dim=0),dim=0).cpu()#.to(device)
 
    tokens_tensor,segments_tensor = tokens_tensor.cpu(), segments_tensor.cpu()
    # tensor.cpu()
    del outputs,tokens_tensor,segments_tensor
            # garbage collect library
    gc.collect()
    torch._C._cuda_emptyCache()
    
    return hidden_states


def load_data(file_path,index):

    
    samples = []
    with open(file_path, 'r' , encoding='latin-1') as file:
        lines = file.readlines()
        max_lines =  len(lines)
        d = 1000+index*1000
        if d > max_lines:
            d = max_lines
        
        for i in range(0+index*1000, d, 4):
            sample = (lines[i].strip(), lines[i+1].strip(), lines[i+2].strip(), lines[i+3].strip())
            samples.append(sample)

    # Example usage
   
    polarities = []
    domain = []
    left_context = []
    target_context = []
    right_context = []
    # Iterate over each tuple in the list
    for idx, (first_element, second_element,third_element,fourth_element) in enumerate(samples):
        left,_,right = split_sentence(first_element,"$T$")
        left_context.append(left)
        target_context.append(second_element)
        right_context.append(right)
        
        # Perform the replacement
        new_first_element = first_element.replace("$T$", second_element)
        # Update the tuple with the replaced string
        samples[idx] = (new_first_element, second_element,third_element)

        
        pol = int(third_element)
        if pol == -1:
            polarities.append([1.0,0.0,0.0])
        elif pol == 0:
            polarities.append([0.0,1.0,0.0])
        elif pol == 1:
            polarities.append([0.0,0.0,1.0])
        else:
            print('error, unkonw polarity')
        
        if fourth_element == 'laptop':
            domain.append([1.0,0.0,0.0])
        elif fourth_element == 'restaurant':
            domain.append([0.0,1.0,0.0])
        elif fourth_element == 'book':   
            domain.append([0.0,0.0,1.0])
        else:
            print(f'error, domain not found: {fourth_element}') 
    # Print the updated list of tuples
    
    sentences = [t[0] for t in samples]
    
    with torch.no_grad():
        left_tokenized_sentences = tokenizer(left_context).get('input_ids')
        right_tokenized_sentences = tokenizer(right_context).get('input_ids')
    
    left_length = [len(l)-1 for l in left_tokenized_sentences]
    right_length = [len(r)-1 for r in right_tokenized_sentences]
    mask_constraints = is_subsequence_stopwords2(sentences)

    polarities = torch.tensor(polarities).cpu()#,device=device)
    domain = torch.tensor(domain).cpu()#,device=device)

    token_ids,segment_ids = get_tokenized(sentences)
    nonzero_counts_per_row = torch.count_nonzero(token_ids, dim=1).cpu().tolist()
    
    
    token_embeddings = get_embeddings(token_ids,segment_ids)
    target_ind = [(l,t- r) for l,r,t in zip(left_length,right_length,nonzero_counts_per_row)]
    
    target_ind = torch.tensor(target_ind)
    del left_length,right_length,nonzero_counts_per_row, left_tokenized_sentences, right_tokenized_sentences,samples
    
    
    return token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,mask_constraints


def split_sentence(sentence, target_string):
    # Find the index of the target string in the sentence
    index = sentence.find(target_string)
    
    if index == -1:
        # If the target string is not found, return None for all contexts
        return None, None, None
    
    # Extract left context
    left_context = sentence[:index]
    
    # Extract target
    target = sentence[index:index+len(target_string)]
    
    # Extract right context
    right_context = sentence[index+len(target_string):]
    
    return left_context.strip(), target.strip(), right_context.strip()

def get_contexts(token_embeddings,target_ind,pad_embedding,segment_ids):
    
    a = torch.sum(segment_ids,dim=1)
    
    target = [token_embeddings[i,j[0]:j[1],:] for i,j in enumerate(target_ind)]
    left = [token_embeddings[i,1:j[0],:] for i,j in enumerate(target_ind)]
    right = [token_embeddings[i,j[1]:a[i]-1,:] for i,j in enumerate(target_ind)]
    
    max_target  = max(max(tensor.size(0) for tensor in target),1)
    max_left = max(1,max(tensor.size(0) for tensor in left))
    max_right = max(max(tensor.size(0) for tensor in right),1)
    
    pad_target = [F.pad(tensor, (0,0,0,max_target-len(tensor))) for tensor in target]
    pad_left = [F.pad(tensor, (0,0,0,max_left-len(tensor))) for tensor in left]
    pad_right = [F.pad(tensor, (0,0,0,max_right-len(tensor))) for tensor in right]


    att_target = []
    for i, x in enumerate(pad_target):
        zeros = torch.all(x==0,dim=1)
        att_target.append(1-zeros.type(torch.int))
        pad_target[i] = pad_embedding.unsqueeze(0).expand(max_target ,pad_embedding.size(0)) * (1-att_target[i]).unsqueeze(1) + pad_target[i] * att_target[i].unsqueeze(1)
   
    

    att_left = []
    for i, x in enumerate(pad_left):
        zeros = torch.all(x==0,dim=1)
        att_left.append(1-zeros.type(torch.int))
        pad_left[i] = pad_embedding.unsqueeze(0).expand(max_left ,pad_embedding.size(0)) * (1-att_left[i]).unsqueeze(1) + pad_left[i] * att_left[i].unsqueeze(1)
        
    att_right = []
    for i, x in enumerate(pad_right):
        zeros = torch.all(x==0,dim=1)
        att_right.append(1-zeros.type(torch.int))
        pad_right[i] = pad_embedding.unsqueeze(0).expand(max_right ,pad_embedding.size(0)) * (1-att_right[i]).unsqueeze(1) + pad_right[i] * att_right[i].unsqueeze(1)

    pad_target = torch.stack(pad_target,dim=0)
    pad_left = torch.stack(pad_left,dim=0)
    pad_right = torch.stack(pad_right,dim=0)
    
    att_target = torch.stack(att_target,dim=0)
    att_left = torch.stack(att_left,dim=0)
    att_right = torch.stack(att_right,dim=0)

    
    del a, target, left, right,zeros
    return pad_target,att_target,pad_left,att_left,pad_right,att_right


    
    
    #sl = torch.cat(sl,dim=0)
    