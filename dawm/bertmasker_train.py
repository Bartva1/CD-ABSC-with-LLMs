import torch
from load_data import CustomDataset,CustomDataset2
from torch.utils.data import DataLoader
from config import *
from bertmasker import SentimentClassifier, SharedPart, PrivatePart, BERTMasker
import torch.nn as nn
from tqdm import tqdm
from evaluation import get_measures
import matplotlib.pyplot as plt

def load_train(domain,target_values):
    out_path = 'train/variables' + domain + '/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    token_ids = torch.load(out_path + 'token_ids.pt')
    segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    target_ind = torch.load(out_path + 'target_ind.pt')
    masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset2(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right,target_tensor_index=8,target_values=target_values),mask_embedding,pad_embedding

def load_test(domain):
    out_path = 'test/variables' + domain + '/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    token_ids = torch.load(out_path + 'token_ids.pt')
    segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    target_ind = torch.load(out_path + 'target_ind.pt')
    masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right)

def load_train2(domain):
    out_path = 'train/variables' + domain + '/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    token_ids = torch.load(out_path + 'token_ids.pt')
    segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    target_ind = torch.load(out_path + 'target_ind.pt')
    masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right),mask_embedding,pad_embedding

def main():
    domain = ''
    targets = [2,1]
    torch.manual_seed(123)
    if torch.cuda.is_available():
        # Set seed for CUDA
        torch.cuda.manual_seed(123)

    dataset,mask_embedding,pad_embedding = load_train(domain,targets)

    data_loader1 = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    domain = '_book'
    dataset,mask_embedding,pad_embedding = load_train2(domain)
    source_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    domain = '_rest'
    
    test_dataset = load_test(domain)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    epochss = 7
    hidden_s = 64
    lr = 5e-05
    weight_decay = 0.0005
    weight_shared = 0.01
    weight_private = weight_shared
    temp =0.5
    weight_sent = 5
    alp = 0.5

    private_part = PrivatePart(hidden_size=hidden_s,temp=temp).to(device)
    shared_part = SharedPart(hidden_size=hidden_s,temp=temp,alpha=alp,masking=0.1).to(device)
    
    sentiment_classifier = SentimentClassifier(hidden_size=hidden_s).to(device)
    
    model = BERTMasker(shared_domain_classifier=shared_part,private_domain_classifier=private_part,sentiment_classifier=sentiment_classifier).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)

    shared_loss_fn = nn.CrossEntropyLoss()
    private_loss_fn = nn.CrossEntropyLoss()
    sentiment_loss_fn = nn.CrossEntropyLoss()
    
    mask_embedding = mask_embedding.to(device)
    i = 0
    train_acc_prev = torch.tensor(0.0,device=device)
    train_acc_prev2 = torch.tensor(0.0,device=device)
    train_acc_prev3 = torch.tensor(0.0,device=device)
    for epoch in range(epochss):
        model.train()
        total_loss = 0.0
        total_shared= 0.0
        total_private = 0.0 
        total_sentiment  = 0.0
        
        train_correct = 0
        train_total = 0.0
        if epoch < 1:
            data_loader = data_loader1
            for name, param in model.named_parameters():
                if 'bert' not in name:
                    if 'shared_lcr' in name or 'private_lcr' in name or 'sentiment_classifier' in name:
                        param.requires_grad = False
                    
        else:
            data_loader = source_loader
            for name, param in model.named_parameters():
                if 'bert' not in name:
                    if 'shared_lcr' in name or 'private_lcr' in name or 'sentiment_classifier' in name:
                        
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        
        # Use tqdm for progress bar
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochss}", unit="batch") as pbar:
            for batch_idx,(hidden_embeddings,_,segments_tensor,polarities,domain,_,_,input_embedding,domain_list,_,_,_,_,_,_) in enumerate(data_loader):
                
                if epoch > 0: 
                    domain_list = torch.zeros(domain_list.size(),device=device,dtype=torch.int64)

                # Zero the gradients
                optimizer.zero_grad()
                hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.to(device),segments_tensor.to(device),polarities.to(device),domain.to(device),domain_list.to(device),input_embedding.to(device)
                #descriptor2 = descriptors.unsqueeze(0).unsqueeze(0).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,descriptor_dimension)
                #hidden_embeddings2 = hidden_embeddings.unsqueeze(2).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,hidden_dim)
                #z = torch.cat((hidden_embeddings2,descriptor2),dim=-1)
                
                # Forward pass
                shared_output,private_output,sentiment_pred,p_embedded_inputs,s_embedded_inputs,p_P,mask_perc = model(hidden_embeddings=hidden_embeddings, input_embedding=input_embedding, mask_embedding=mask_embedding, segments_tensor=segments_tensor, domain_list=domain_list)
                i+=1
                #print(torch.sum(s_P*segments_tensor,dim=-1))
                #print(torch.sum(p_P*ssegments_tensor,dim=-1))
        
                #print(p_embedded_inputs.size())
                
                if epoch <1:
                    shared_loss = shared_loss_fn(shared_output, domain_list)
                    private_loss = private_loss_fn(private_output, domain_list)
                    epoch_loss =  weight_shared*shared_loss + weight_private* private_loss
                  
                    total_loss += epoch_loss.item()
                    total_shared += shared_loss.item()
                    total_private += private_loss.item()

                elif epoch <10:
                    sentiment_loss = sentiment_loss_fn(sentiment_pred,torch.argmax(polarities,dim=1))
                    epoch_loss = weight_sent* sentiment_loss

                    total_loss += epoch_loss.item() 
                    
                else:
                    shared_loss = shared_loss_fn(shared_output, domain_list)
                    private_loss = private_loss_fn(private_output, domain_list)
                    sentiment_loss = sentiment_loss_fn(sentiment_pred,torch.argmax(polarities,dim=1))

                    epoch_loss = weight_shared*shared_loss + weight_private* private_loss + weight_sent* sentiment_loss

                    total_loss += epoch_loss.item()
                    total_shared += shared_loss.item()
                    total_private += private_loss.item()
                    total_sentiment += sentiment_loss.item()
                #print(epoch_loss)
                # Backward pass
                
                epoch_loss.backward(retain_graph=True)
                
                #train_acc += torchmetrics.functional.accuracy(torch.argmax(nn.functional.softmax(sentiment_pred,dim=-1),dim=1), torch.argmax(polarities,dim=1),task='multiclass',num_classes=num_polarities)
                
                train_correct += torch.sum((torch.argmax(nn.functional.softmax(sentiment_pred,dim=-1),dim=1) == torch.argmax(polarities,dim=1)).type(torch.int)).item()
                train_total += polarities.size(0)
                # Update weights
                optimizer.step()
                hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.cpu(),segments_tensor.cpu(),polarities.cpu(),domain.cpu(),domain_list.cpu(),input_embedding.cpu()
                
                
                # Update progress bar
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                pbar.update(1)
                #print(descriptors)
                
                if i ==0:
                    
                    for name, param in model.named_parameters():
                        #if 'descr' in name:
                        #    print(param.grad)
                    
                        if param.grad == None:
                            print(name, param.grad)
                        ##else:
                        #if 0 in param.grad:
                        #    print(param.grad)
                    print(epoch_loss)
                
        i = 0
        
        
        print(f'Epoch [{epoch+1}/{epochss}], Total Loss: {total_loss:.4f}, shared loss {total_shared:.4f}, private loss: {total_private:.4f}, sentiment loss: {total_sentiment:.4f}')

        train_acc = torch.tensor(train_correct / train_total,device=device)
        print(f'acc: {train_acc}')
        train_acc_prev3 = train_acc_prev2
        train_acc_prev2 = train_acc_prev
        train_acc_prev = train_acc
        
        #if torch.max(train_acc_prev2,train_acc_prev) - train_acc_prev3 < eps:
        #    break
        
    for current_domain in ['Rest-laptop']:
        
        model.eval()
        

        with torch.no_grad():
            for batch_idx,(hidden_embeddings,_,segments_tensor,polarities,domain,_,_,input_embedding,domain_list,_,_,_,_,_,_) in enumerate(test_dataloader):
                hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.to(device),segments_tensor.to(device),polarities.to(device),domain.to(device),domain_list.to(device),input_embedding.to(device)
                #descriptor2 = descriptors.unsqueeze(0).unsqueeze(0).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,descriptor_dimension)
                #hidden_embeddings2 = hidden_embeddings.unsqueeze(2).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,hidden_dim)
                #z = torch.cat((hidden_embeddings2,descriptor2),dim=-1)
                domain_list = torch.ones(domain_list.size(),device=device,dtype=torch.int64)
                # Forward pass
                shared_output,private_output,sentiment_pred,p_embedded_inputs,s_embedded_inputs,p_P,mask_perc = model(hidden_embeddings=hidden_embeddings, input_embedding=input_embedding, mask_embedding=mask_embedding, segments_tensor=segments_tensor, domain_list=domain_list)
                
                polarities = torch.argmax(polarities,dim=1)
                pred = torch.argmax(nn.functional.softmax(sentiment_pred,dim=-1),dim=1)
                shared_pred = torch.argmax(nn.functional.softmax(shared_output,dim=-1),dim=1)
                private_pred = torch.argmax(nn.functional.softmax(private_output,dim=-1),dim=1)
                if batch_idx == 0:
                    y_test = polarities
                    y_pred = pred
                    domain_test = domain_list
                    domain_pred_shared = shared_pred
                    domain_pred_private = private_pred
                    total_mask = mask_perc
                    
                else:
                    y_test = torch.cat((y_test,polarities))
                    y_pred = torch.cat((y_pred,pred))
                    domain_test = torch.cat((domain_test,domain_list))
                    domain_pred_shared = torch.cat((domain_pred_shared,shared_pred))
                    domain_pred_private = torch.cat((domain_pred_private,private_pred))
                    total_mask = torch.cat((total_mask,mask_perc))
                    
                hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.cpu(),segments_tensor.cpu(),polarities.cpu(),domain.cpu(),domain_list.cpu(),input_embedding.cpu()

        neg_indices = torch.nonzero(y_test == 0, as_tuple=True)
        neutral_indices = torch.nonzero(y_test == 1, as_tuple=True)
        pos_indices = torch.nonzero(y_test == 2, as_tuple=True)

        measures = get_measures(y_test=y_test.cpu().numpy(),y_pred=y_pred.cpu().numpy(),samplewise='all')
        neg_measures = get_measures(y_test=y_test[neg_indices].cpu().numpy(),y_pred=y_pred[neg_indices].cpu().numpy())
        neutral_measures = get_measures(y_test=y_test[neutral_indices].cpu().numpy(),y_pred=y_pred[neutral_indices].cpu().numpy())
        pos_measures = get_measures(y_test=y_test[pos_indices].cpu().numpy(),y_pred=y_pred[pos_indices].cpu().numpy())

        shared_measures = get_measures(y_test=domain_test.cpu().numpy(),y_pred=domain_pred_shared.cpu().numpy())
        private_measures = get_measures(y_test=domain_test.cpu().numpy(),y_pred=domain_pred_private.cpu().numpy())
        
        print(current_domain)
        print(f'measures: {measures}')
        print(f'neg measures: {neg_measures}')
        print(f'neutral measures: {neutral_measures}')
        print(f'pos measures: {pos_measures}')

        print(f'shared measures: {shared_measures}')
        print(f'private measures: {private_measures}')

        total_mask = total_mask.cpu().numpy()
        plt.figure()
        # Create histogram with specified range
        plt.hist(total_mask * 100, bins = 30, range=(0, 100),edgecolor = 'black',color = 'grey')
        plt.title(current_domain)
        # Add title and labels
        plt.xlabel('Percentage masked')
        plt.ylabel('Number of reviews')

        # Show plot
    plt.show()

    pbar.close()
    return measures,neg_measures,neutral_measures,pos_measures
    
if __name__ == '__main__':

    main()