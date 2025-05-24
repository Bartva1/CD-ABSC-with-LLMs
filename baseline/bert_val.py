import torch
from load_data import CustomDataset
from torch.utils.data import DataLoader
from config import *
from bert import SentimentClassifier
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
import optuna

def load_train(domain):
    """
    Load training data for a specified domain.
    
    Args:
        domain (str): The domain for which to load the training data.
    
    Returns:
        tuple: A tuple containing the training dataset, mask embeddings, and pad embeddings.
    """
    out_path = 'train_small/variables' + domain + '/'
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

def load_val(domain):
    """
    Load validation data for a specified domain.
    
    Args:
        domain (str): The domain for which to load the validation data.
    
    Returns:
        CustomDataset: The validation dataset.
    """
    out_path = 'val/variables' + domain + '/'
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

def main(trial, validation_dataloader, data_loader, mask_embedding, pad_embedding):
    """
    Train a BERT model with a classification layer and evaluate it on the validation set.
    
    Args:
        trial (optuna.trial): Optuna trial for hyperparameter optimization.
        validation_dataloader (DataLoader): DataLoader for the validation set.
        data_loader (DataLoader): DataLoader for the training set.
        mask_embedding (tensor): Mask embedding tensor.
        pad_embedding (tensor): Pad embedding tensor.
    
    Returns:
        float: The validation accuracy of the model.
    """
    # Suggest hyperparameters for the trial
    hidden_s = trial.suggest_categorical("hidden_s", [64, 128, 256])
    lr = trial.suggest_categorical("lr", [0.01, 0.001, 0.0001, 0.0005, 0.005, 0.00005, 0.00001])
    weight_decay = trial.suggest_categorical("weight_decay", [0.01, 0.001, 0.0001, 0.0005, 0.005])
    epochss = 50
    
    model = SentimentClassifier(hidden_size=hidden_s).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sentiment_loss_fn = nn.CrossEntropyLoss()

    train_acc_prev = torch.tensor(0.0, device=device)
    train_acc_prev2 = torch.tensor(0.0, device=device)
    train_acc_prev3 = torch.tensor(0.0, device=device)
    
    for epoch in range(epochss):
        model.train()
        total_loss = 0.0
        total_shared = 0.0
        total_private = 0.0 
        total_sentiment  = 0.0
        train_correct = 0
        train_total = 0.0
  
        # Use tqdm for progress bar
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochss}", unit="batch") as pbar:
            for batch_idx, (hidden_embeddings, _, segments_tensor, polarities, domain, _, _, input_embedding, domain_list, _, _, _, _, _, _) in enumerate(data_loader):
                optimizer.zero_grad()
                hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = hidden_embeddings.to(device), segments_tensor.to(device), polarities.to(device), domain.to(device), domain_list.to(device), input_embedding.to(device)
                
                s = torch.sum(segments_tensor, dim=1).to(device) - 1
                zeros = torch.zeros(input_embedding.size(0), MAX_LENGTH).to(device)
                zeros.scatter_(1, s.unsqueeze(1), 1).scatter_(1, torch.zeros_like(zeros[:, :1]).to(device).long(), 1)

                # Calculate the representation by summing over the token embeddings
                rep = torch.sum(hidden_embeddings * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1), dim=1) / torch.sum(segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1), dim=1)
                
                # Forward pass
                output = model(representation=rep)
                epoch_loss = sentiment_loss_fn(output, torch.argmax(polarities, dim=1))
                total_loss += epoch_loss.item()
                
                # Backward pass
                epoch_loss.backward(retain_graph=True)
                
                # Update training accuracy
                train_correct += torch.sum((torch.argmax(nn.functional.softmax(output, dim=-1), dim=1) == torch.argmax(polarities, dim=1)).type(torch.int)).item()
                train_total += polarities.size(0)
                optimizer.step()
                hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = hidden_embeddings.cpu(), segments_tensor.cpu(), polarities.cpu(), domain.cpu(), domain_list.cpu(), input_embedding.cpu()
                
                # Update progress bar
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
            s = torch.sum(segments_tensor, dim=1).to(device) - 1
            zeros = torch.zeros(input_embedding.size(0), MAX_LENGTH).to(device)
            zeros.scatter_(1, s.unsqueeze(1), 1).scatter_(1, torch.zeros_like(zeros[:, :1]).to(device).long(), 1)
            rep = torch.sum(hidden_embeddings * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1), dim=1) / torch.sum(segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1), dim=1)
            
            output = model(representation=rep)
            polarities = torch.argmax(polarities, dim=1)
            pred = torch.argmax(nn.functional.softmax(output, dim=-1), dim=1)
            test_correct += torch.sum((pred == polarities).type(torch.int)).item()
            test_total += polarities.size(0)
            f1 += torchmetrics.functional.f1_score(pred, polarities, task='multiclass', num_classes=num_polarities, average='macro')
            hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = hidden_embeddings.cpu(), segments_tensor.cpu(), polarities.cpu(), domain.cpu(), domain_list.cpu(), input_embedding.cpu()

    print(f'Epoch [{epoch+1}/{epochss}], Accuracy: {test_correct / test_total:.4f}, F1 Score: {f1 / len(validation_dataloader):.4f}')
    acc = test_correct / test_total
    return acc

if __name__ == '__main__':
    """
    Main execution block to load data, initialize parameters, and train/evaluate the model with hyperparameter optimization.
    """
    domain = '_laptop'
    
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    dataset, mask_embedding, pad_embedding = load_train(domain)
    val_dataset = load_val(domain)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print('Data loaded')
 
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))
    study.optimize(lambda trial: main(trial, validation_dataloader, data_loader, mask_embedding, pad_embedding), n_trials=15)

    # Get the best hyperparameters and results
    best_params = study.best_params
    best_loss = study.best_value
    print(best_params)
    print(best_loss)
