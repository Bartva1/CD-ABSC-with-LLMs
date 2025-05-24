import torch
from load_data import CustomDataset
from torch.utils.data import DataLoader
from config import *
from lcr_rot_hopplusplus import LCRRotHopPlusPlus
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
import optuna

def load_train(domain):
    out_path = 'train_small/variables_' + domain + '/'
    #token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    #token_ids = torch.load(out_path + 'token_ids.pt')
    #segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    #target_ind = torch.load(out_path + 'target_ind.pt')
    #masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    #input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    #domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    #mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    #pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right)

def load_val(domain):
    out_path = 'val/variables_' + domain + '/'
    #token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    #token_ids = torch.load(out_path + 'token_ids.pt')
    #segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    #target_ind = torch.load(out_path + 'target_ind.pt')
    #masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    #input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    #domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    #mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    #pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right)


def main(trial, val_data_loader,data_loader):
    


    lr = trial.suggest_categorical("lr", [0.01,0.001,0.0001,0.0005,0.005,0.00005,0.00001])
    weight_decay = trial.suggest_categorical("weight_decay", [0.01,0.001,0.0001,0.0005,0.005])
    #epochss = trial.suggest_int("epochss",5,15,step=2)
    #dropout1 = trial.suggest_categorical('dropout1',[0.2,0.3,0.4,0.5]) 
    #dropout2 = trial.suggest_categorical('dropout2',[0.2,0.3,0.4,0.5]) 
    epochss = 50
    model = LCRRotHopPlusPlus().to(device)

    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    
    i = 0
    train_acc_prev = torch.tensor(0.0,device=device)
    train_acc_prev2 = torch.tensor(0.0,device=device)
    train_acc_prev3 = torch.tensor(0.0,device=device)
    for epoch in range(epochss):
        model.train()
        total_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        train_total = 0.0
        # Use tqdm for progress bar
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochss}", unit="batch") as pbar:
            for batch_idx,(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right) in enumerate(data_loader):
                
                # Zero the gradients
                optimizer.zero_grad()
                polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right = polarities.to(device),pad_target.to(device),att_target.to(device),pad_left.to(device),att_left.to(device),pad_right.to(device),att_right.to(device)
                
                # Forward pass
                output = model(left = pad_left,target = pad_target,right = pad_right,att_left = att_left,att_target = att_target,att_right = att_right)
                i+=1
                loss = loss_fn(output, torch.argmax(polarities,dim=1))
                #accuracy += torchmetrics.functional.accuracy( torch.argmax(nn.functional.softmax(output,dim=-1),dim=1), torch.argmax(polarities,dim=1),task='multiclass',num_classes=num_polarities)
                #train_acc += torchmetrics.functional.accuracy(torch.argmax(nn.functional.softmax(output,dim=-1),dim=1), torch.argmax(polarities,dim=1),task='multiclass',num_classes=num_polarities)
                train_correct += torch.sum((torch.argmax(nn.functional.softmax(output,dim=-1),dim=1) == torch.argmax(polarities,dim=1)).type(torch.int)).item()
                train_total += polarities.size(0)
                loss.backward()#retain_graph=True)
                total_loss += loss.item()
                # Update weights
                optimizer.step()
                polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right = polarities.cpu(),pad_target.cpu(),att_target.cpu(),pad_left.cpu(),att_left.cpu(),pad_right.cpu(),att_right.cpu()
                # Update progress bar
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                pbar.update(1)
                
                if i == 0:
                    
                    for name, param in model.named_parameters():
                        if param.grad == None:
                            print(name, param.grad)
            i = 0
        print(f'Epoch [{epoch+1}/{epochss}], Loss: {total_loss:.4f}, acc: {train_acc / len(data_loader):.4f}, correct: {train_correct}, trst: {train_correct / train_total}')
        train_acc = torch.tensor(train_correct / train_total,device=device)
        train_acc_prev3 = train_acc_prev2
        train_acc_prev2 = train_acc_prev
        train_acc_prev = train_acc
        

        accuracy = 0.0
        f1 = 0.0
        model.eval()
        test_correct = 0.0
        test_total = 0.0
        
        with torch.no_grad():
            for batch_idx,(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right) in enumerate(val_data_loader):
                polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right = polarities.to(device),pad_target.to(device),att_target.to(device),pad_left.to(device),att_left.to(device),pad_right.to(device),att_right.to(device)
                
                # Forward pass
                output = model(left = pad_left,target = pad_target,right = pad_right,att_left = att_left,att_target = att_target,att_right = att_right)
                i+=1
                
                pred = torch.argmax(nn.functional.softmax(output,dim=-1),dim=1)
                accuracy += torchmetrics.functional.accuracy(pred, torch.argmax(polarities,dim=1),task='multiclass',num_classes=num_polarities)

                test_correct += torch.sum((torch.argmax(nn.functional.softmax(output,dim=-1),dim=1) == torch.argmax(polarities,dim=1)).type(torch.int)).item()
                test_total += polarities.size(0)

                f1 += torchmetrics.functional.f1_score(pred,torch.argmax(polarities,dim=1),task='multiclass',num_classes=num_polarities,average='macro')

                polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right = polarities.cpu(),pad_target.cpu(),att_target.cpu(),pad_left.cpu(),att_left.cpu(),pad_right.cpu(),att_right.cpu()
        print(f'Epoch [{epoch+1}/{epochss}], acc: {accuracy / len(val_data_loader):.4f}, f1: {f1 / len(val_data_loader):.4f}, acc: {test_correct / test_total}')
        acc = test_correct / test_total
        if torch.max(train_acc_prev2,train_acc_prev) - train_acc_prev3 < eps:
            break
    pbar.close()
    return acc
    
    
if __name__ == '__main__':

    domain = 'rest'
    dataset = load_train(domain)

    val_data = load_val(domain)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        # Set seed for CUDA
        torch.cuda.manual_seed(123)

    print('loaded data')
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))
    study.optimize(lambda trial: main(trial, val_data_loader,data_loader), n_trials=15)

    # Get the best hyperparameters and results
    best_params = study.best_params
    best_loss = study.best_value
    print(best_params)
    print(best_loss)

    