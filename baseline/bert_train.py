import torch
from load_data import CustomDataset
from torch.utils.data import DataLoader
from config import *
from bert import SentimentClassifier
import torch.nn as nn
from tqdm import tqdm
from evaluation import get_measures

def load_train(domain):
    """
    Load training data for a specified domain.
    
    Args:
        domain (str): The domain for which to load the training data.
    
    Returns:
        tuple: A tuple containing the training dataset, mask embeddings, and pad embeddings.
    """
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
    
    return CustomDataset(token_embeddings, token_ids, segment_ids, polarities, domain, target_ind, masking_constraints, input_embeddings, domain_list, pad_target, att_target, pad_left, att_left, pad_right, att_right), mask_embedding, pad_embedding

def load_test(domain):
    """
    Load test data for a specified domain.
    
    Args:
        domain (str): The domain for which to load the test data.
    
    Returns:
        CustomDataset: The test dataset.
    """
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

    return CustomDataset(token_embeddings, token_ids, segment_ids, polarities, domain, target_ind, masking_constraints, input_embeddings, domain_list, pad_target, att_target, pad_left, att_left, pad_right, att_right)

def main(param, validation_dataloader, data_loader):
    """
    Train a BERT model with a classification layer and evaluate it on the validation set.
    
    Args:
        param (dict): Dictionary of training parameters.
        validation_dataloader (DataLoader): DataLoader for the validation set.
        data_loader (DataLoader): DataLoader for the training set.
    
    Returns:
        tuple: A tuple containing overall measures, negative class measures, neutral class measures, and positive class measures.
    """
    hidden_s = param['hidden_s']
    lr = param['lr']
    weight_decay = param['weight_decay']
    epochss = 50
    
    # Initialize the sentiment classifier model and optimizer
    model = SentimentClassifier(hidden_size=hidden_s).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sentiment_loss_fn = nn.CrossEntropyLoss()

    train_acc_prev = torch.tensor(0.0, device=device)
    train_acc_prev2 = torch.tensor(0.0, device=device)
    train_acc_prev3 = torch.tensor(0.0, device=device)
    
    for epoch in range(epochss):
        model.train()
        total_loss = 0.0
        train_correct = 0
        train_total = 0.0
  
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochss}", unit="batch") as pbar:
            for batch_idx, (hidden_embeddings, _, segments_tensor, polarities, domain, _, _, input_embedding, domain_list, _, _, _, _, _, _) in enumerate(data_loader):
                optimizer.zero_grad()
                hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = hidden_embeddings.to(device), segments_tensor.to(device), polarities.to(device), domain.to(device), domain_list.to(device), input_embedding.to(device)
                
                # Calculate the index of the last token in each sequence
                s = torch.sum(segments_tensor, dim=1).to(device) - 1
                zeros = torch.zeros(input_embedding.size(0), MAX_LENGTH).to(device)
                
                # Create a mask for the first and last tokens in each sequence
                zeros.scatter_(1, s.unsqueeze(1), 1).scatter_(1, torch.zeros_like(zeros[:, :1]).to(device).long(), 1)
                
                # Use the first token's embedding as the sentence representation
                rep = hidden_embeddings[:, 0, :]
                output = model(representation=rep)
                
                # Calculate the loss and backpropagate
                epoch_loss = sentiment_loss_fn(output, torch.argmax(polarities, dim=1))
                total_loss += epoch_loss.item()
                epoch_loss.backward(retain_graph=True)
                
                # Update the training accuracy
                train_correct += torch.sum((torch.argmax(nn.functional.softmax(output, dim=-1), dim=1) == torch.argmax(polarities, dim=1)).type(torch.int)).item()
                train_total += polarities.size(0)
                optimizer.step()
                
                hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = hidden_embeddings.cpu(), segments_tensor.cpu(), polarities.cpu(), domain.cpu(), domain_list.cpu(), input_embedding.cpu()
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                pbar.update(1)
        
        print(f'Epoch [{epoch+1}/{epochss}], Total Loss: {total_loss:.4f}')
        train_acc = torch.tensor(train_correct / train_total, device=device)
        print(f'Accuracy: {train_acc}')
        train_acc_prev3 = train_acc_prev2
        train_acc_prev2 = train_acc_prev
        train_acc_prev = train_acc
        
        # Early stopping condition based on training accuracy
        if torch.max(train_acc_prev2, train_acc_prev) - train_acc_prev3 < eps:
            break

    model.eval()
    f1 = 0.0
    test_correct = 0.0
    test_total = 0.0

    with torch.no_grad():
        for batch_idx, (hidden_embeddings, _, segments_tensor, polarities, domain, _, _, input_embedding, domain_list, _, _, _, _, _, _) in enumerate(validation_dataloader):
            hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = hidden_embeddings.to(device), segments_tensor.to(device), polarities.to(device), domain.to(device), domain_list.to(device), input_embedding.to(device)
            
            # Calculate the index of the last token in each sequence
            s = torch.sum(segments_tensor, dim=1).to(device) - 1
            zeros = torch.zeros(input_embedding.size(0), MAX_LENGTH).to(device)
            
            # Create a mask for the first and last tokens in each sequence
            zeros.scatter_(1, s.unsqueeze(1), 1).scatter_(1, torch.zeros_like(zeros[:, :1]).to(device).long(), 1)
            
            # Use the first token's embedding as the sentence representation
            rep = hidden_embeddings[:, 0, :]
            output = model(representation=rep)

            polarities = torch.argmax(polarities, dim=1)
            pred = torch.argmax(nn.functional.softmax(output, dim=-1), dim=1)

            # Collect predictions and true labels for evaluation
            if batch_idx == 0:
                y_test = polarities
                y_pred = pred
            else:
                y_test = torch.cat((y_test, polarities))
                y_pred = torch.cat((y_pred, pred))
            
            hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = hidden_embeddings.cpu(), segments_tensor.cpu(), polarities.cpu(), domain.cpu(), domain_list.cpu(), input_embedding.cpu()
    
    # Separate predictions by class for detailed evaluation
    neg_indices = torch.nonzero(y_test == 0, as_tuple=True)
    neutral_indices = torch.nonzero(y_test == 1, as_tuple=True)
    pos_indices = torch.nonzero(y_test == 2, as_tuple=True)

    measures = get_measures(y_test.cpu(), y_pred.cpu())
    neg_measures = get_measures(y_test[neg_indices].cpu(), y_pred[neg_indices].cpu())
    neutral_measures = get_measures(y_test[neutral_indices].cpu(), y_pred[neutral_indices].cpu())
    pos_measures = get_measures(y_test[pos_indices].cpu(), y_pred[pos_indices].cpu())
    
    # Output evaluation metrics for each class
    print(f'Negative Measures: {neg_measures}')
    print(f'Neutral Measures: {neutral_measures}')
    print(f'Positive Measures: {pos_measures}')

    return measures, neg_measures, neutral_measures, pos_measures

if __name__ == '__main__':
    """
    Main execution block to load data, initialize parameters, and train/evaluate the model.
    """
    # Domain is either [_rest,_laptop,_book]
    domain = '_rest'
    
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    dataset, mask_embedding, pad_embedding = load_train(domain)
    test_dataset = load_test(domain)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print('Data loaded')
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    param = {'hidden_s': 256, 'lr': 5e-05, 'weight_decay': 0.001}
    measures = main(param, test_dataloader, data_loader)
