import torch
from load_data import load_data,get_contexts
from config import *
import math
from tqdm import tqdm
import os 
print("Selected device:", device)

domains = ["restaurant", "book"]
train_test = ["train"]

CHUNK_SIZE = 999  

def save_tensor_in_chunks(tensor, out_path, name, chunk_size=CHUNK_SIZE):
    total = len(tensor)
    for i in range(0, total, chunk_size):
        chunk = tensor[i:i+chunk_size]
        torch.save(chunk, os.path.join(out_path, f"{name}_{i//chunk_size}.pt"))

for domain_name in domains:
    for phase in train_test:

        year = 2019 if domain_name == 'book' else 2014
        torch.manual_seed(123)
        if torch.cuda.is_available():
            # Set seed for CUDA
            torch.cuda.manual_seed(123)

        file_path = f"data_out/{domain_name}/raw_data_{domain_name}_{phase}_{year}.txt"  # Replace 'your_text_file.txt' with the path to your text file

        with open(file_path, 'r',encoding='latin-1') as file:
            # Initialize a counter for lines
            line_count = 0
            
            # Iterate through each line in the file
            for line in file:
                line_count += 1
        line_count =  int(math.ceil(line_count / 999))
        print(line_count)
        for i in tqdm(range(line_count)):
            if i == 0:
                token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints = load_data(file_path,i, domain_name)
                token_embeddings=token_embeddings.cpu()
                token_ids=token_ids.cpu()
                segment_ids=segment_ids.cpu()
                polarities=polarities.cpu()
                domain=domain.cpu()
                target_ind = target_ind.cpu()
                masking_constraints = masking_constraints.cpu()
                
                torch.cuda.empty_cache() 
            else:
                
                token_embeddings_it,token_ids_it,segment_ids_it,polarities_it,domain_it,target_ind_it,masking_constraints_it = load_data(file_path,i, domain_name)
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

        
        out_path = f"{phase}/variables_{domain_name}/"
        os.makedirs(out_path, exist_ok=True)

        if phase == "train": 
            out_path = f"{phase}/variables_{domain_name}/chunked/"
            os.makedirs(out_path, exist_ok=True)
            save_tensor_in_chunks(token_embeddings, out_path, 'token_embeddings')
            save_tensor_in_chunks(token_ids, out_path, 'token_ids')
            save_tensor_in_chunks(segment_ids, out_path, 'segment_ids')
            save_tensor_in_chunks(polarities, out_path, 'polarities')
            save_tensor_in_chunks(domain, out_path, 'domain')
            save_tensor_in_chunks(target_ind, out_path, 'target_ind')
            save_tensor_in_chunks(masking_constraints, out_path, 'masking_constraints')
            save_tensor_in_chunks(input_embeddings, out_path, 'input_embeddings')
            save_tensor_in_chunks(domain_list, out_path, 'domain_list')
            save_tensor_in_chunks(pad_target, out_path, 'pad_target')
            save_tensor_in_chunks(att_target, out_path, 'att_target')
            save_tensor_in_chunks(pad_left, out_path, 'pad_left')
            save_tensor_in_chunks(att_left, out_path, 'att_left')
            save_tensor_in_chunks(pad_right, out_path, 'pad_right')
            save_tensor_in_chunks(att_right, out_path, 'att_right')
        else:
            torch.save(token_embeddings, os.path.join(out_path, 'token_embeddings.pt'))
            torch.save(token_ids, os.path.join(out_path, 'token_ids.pt'))
            torch.save(segment_ids,os.path.join(out_path, 'segment_ids.pt'))
            torch.save(polarities, os.path.join(out_path, 'polarities.pt'))
            torch.save(domain,os.path.join(out_path, 'domain.pt'))
            torch.save(target_ind,os.path.join(out_path, 'target_ind.pt'))
            torch.save(masking_constraints,os.path.join(out_path, 'masking_constraints.pt'))
            torch.save(input_embeddings,os.path.join(out_path, 'input_embeddings.pt'))
            torch.save(domain_list,os.path.join(out_path, 'domain_list.pt'))
            torch.save(pad_target,os.path.join(out_path, 'pad_target.pt'))
            torch.save(att_target, os.path.join(out_path, 'att_target.pt'))
            torch.save(pad_left, os.path.join(out_path, 'pad_left.pt'))
            torch.save(att_left, os.path.join(out_path, 'att_left.pt'))
            torch.save(pad_right, os.path.join(out_path, 'pad_right.pt'))
            torch.save(att_right, os.path.join(out_path, 'att_right.pt'))

        # Always save these (they are small)
        torch.save(mask_embedding,os.path.join(out_path, 'mask_embedding.pt'))
        torch.save(pad_embedding, os.path.join(out_path, 'pad_embedding.pt'))