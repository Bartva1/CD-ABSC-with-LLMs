import torch
from load_data import load_data,get_contexts
from config import *
import math
print("Selected device:", device)

domain = 'all'
year = ''
torch.manual_seed(123)
if torch.cuda.is_available():
    # Set seed for CUDA
    torch.cuda.manual_seed(123)

file_path = 'train/domain_raw_data_' + domain + year + '.txt'  # Replace 'your_text_file.txt' with the path to your text file

with open(file_path, 'r',encoding='latin-1') as file:
    # Initialize a counter for lines
    line_count = 0
    
    # Iterate through each line in the file
    for line in file:
        line_count += 1
line_count =  int(math.ceil(line_count / 1000))
print(line_count)
for i in range(line_count):
    print(i)
    if i == 0:
        token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints = load_data(file_path,i)
        token_embeddings=token_embeddings.cpu()
        token_ids=token_ids.cpu()
        segment_ids=segment_ids.cpu()
        polarities=polarities.cpu()
        domain=domain.cpu()
        target_ind = target_ind.cpu()
        masking_constraints = masking_constraints.cpu()
        
        torch.cuda.empty_cache() 
    else:
        
        token_embeddings_it,token_ids_it,segment_ids_it,polarities_it,domain_it,target_ind_it,masking_constraints_it = load_data(file_path,i)
        token_embeddings_it = token_embeddings_it.cpu()
        token_ids_it=token_ids_it.cpu()
        segment_ids_it=segment_ids_it.cpu()
        polarities_it=polarities_it.cpu()
        domain_it=domain_it.cpu()
        target_ind_it = target_ind_it.cpu()
        masking_constraints_it = masking_constraints_it.cpu()
        
        token_embeddings = torch.cat((token_embeddings,token_embeddings_it),dim=0)
        token_ids = torch.cat((token_ids,token_ids_it),dim=0)
        segment_ids = torch.cat((segment_ids,segment_ids_it),dim=0)
        polarities = torch.cat((polarities,polarities_it),dim=0)
        domain = torch.cat((domain,domain_it),dim=0)
        target_ind = torch.cat((target_ind,target_ind_it),dim=0)
        masking_constraints = torch.cat((masking_constraints,masking_constraints_it),dim=0)
        
        torch.cuda.empty_cache() 
        
        
        del token_embeddings_it, token_ids_it, segment_ids_it ,polarities_it,domain_it,target_ind_it,masking_constraints_it

model_bert2 = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)#.to(device)
for param in model_bert2.parameters():
    param.requires_grad = False
vocab = {id: model_bert2.get_input_embeddings()(torch.tensor(id))  for token, id in tokenizer.get_vocab().items()}



# Convert the token_ids tensor to a list of lists
token_ids_list = token_ids.tolist()

# Use list comprehension to create a list of embeddings for each token ID
# Then stack them along a new dimension to create a tensor
input_embeddings = torch.stack([vocab[token_id] for row in token_ids_list for token_id in row])#.to(device)

# Reshape the tensor to match the desired shape (8x87x768)
input_embeddings = input_embeddings.view(len(token_ids), MAX_LENGTH, hidden_dim)

domain_list = torch.eq(domain, 1.0)#.to(device)

# Get the indices where the condition is true
domain_list = torch.nonzero(domain_list)

domain_list =domain_list[:,-1]#.tolist()

pad_target,att_target,pad_left,att_left,pad_right,att_right = get_contexts(token_embeddings,target_ind,vocab[0],segment_ids)

mask_embedding = vocab[103]
pad_embedding = vocab[0]

out_path = 'train/variables/'
torch.save(token_embeddings,out_path + 'token_embeddings.pt')
torch.save(token_ids,out_path + 'token_ids.pt')
torch.save(segment_ids,out_path + 'segment_ids.pt')
torch.save(polarities,out_path + 'polarities.pt')
torch.save(domain,out_path + 'domain.pt')
torch.save(target_ind,out_path + 'target_ind.pt')
torch.save(masking_constraints,out_path + 'masking_constraints.pt')
torch.save(input_embeddings,out_path + 'input_embeddings.pt')
torch.save(domain_list,out_path + 'domain_list.pt')

torch.save(pad_target,out_path + 'pad_target.pt')
torch.save(att_target,out_path + 'att_target.pt')
torch.save(pad_left,out_path + 'pad_left.pt')
torch.save(att_left,out_path + 'att_left.pt')
torch.save(pad_right,out_path + 'pad_right.pt')
torch.save(att_right,out_path + 'att_right.pt')

torch.save(mask_embedding,out_path + 'mask_embedding.pt')
torch.save(pad_embedding,out_path + 'pad_embedding.pt')
