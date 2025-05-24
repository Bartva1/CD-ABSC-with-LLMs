import torch
from load_data import CustomDataset
from torch.utils.data import DataLoader
from config import *
from lcr_rot_hopplusplus import LCRRotHopPlusPlus
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
from evaluation import get_measures

def load_train(domain):
    out_path = 'train/variables_' + domain + '/'
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

def load_test(domain):
    out_path = 'test/variables_' + domain + '/'
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


def main():
    domain = 'laptop'
    dataset = load_train(domain)
    domain = 'rest'
    test_data = load_test(domain)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        # Set seed for CUDA
        torch.cuda.manual_seed(123)
    
    param = {'lr': 0.0005,'weight_decay': 0.01}

    print('loaded data')
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LCRRotHopPlusPlus().to(device)


    #optimizer = torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=0.01)
    optimizer = torch.optim.AdamW(model.parameters(),lr=param['lr'],weight_decay=param['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    num_epochs = 7
    i = 0

    train_acc_prev = torch.tensor(0.0,device=device)
    train_acc_prev2 = torch.tensor(0.0,device=device)
    train_acc_prev3 = torch.tensor(0.0,device=device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_acc = 0.0
        train_correct = 0
        train_total =0.0
        # Use tqdm for progress bar
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx,(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right) in enumerate(data_loader):
                
                # Zero the gradients
                optimizer.zero_grad()
                polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right = polarities.to(device),pad_target.to(device),att_target.to(device),pad_left.to(device),att_left.to(device),pad_right.to(device),att_right.to(device)
            
                # Forward pass
                output = model(left = pad_left,target = pad_target,right = pad_right,att_left = att_left,att_target = att_target,att_right = att_right)
                i+=1
                loss = loss_fn(output, torch.argmax(polarities,dim=1))
                train_acc += torchmetrics.functional.accuracy(torch.argmax(nn.functional.softmax(output,dim=-1),dim=1), torch.argmax(polarities,dim=1),task='multiclass',num_classes=num_polarities)
                train_total += polarities.size(0)
                train_correct += torch.sum((torch.argmax(nn.functional.softmax(output,dim=-1),dim=1) == torch.argmax(polarities,dim=1)).type(torch.int)).item()
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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, acc: {train_acc / len(data_loader):.4f}, correct: {train_correct}')
        train_acc = torch.tensor(train_correct / train_total,device=device)
        print(f'acc: {train_acc}')

        train_acc_prev3 = train_acc_prev2
        train_acc_prev2 = train_acc_prev
        train_acc_prev = train_acc
        
        if torch.max(train_acc_prev2,train_acc_prev) - train_acc_prev3 < eps:
            break

    for j in range (3):
        if j == 0:
            test_data = load_test('laptop')
            test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        if j == 1:
            test_data = load_test('rest')
            test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        if j == 2:
            test_data = load_test('book')
            test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        model.eval()
        
        with torch.no_grad():
            for batch_idx,(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right) in enumerate(test_data_loader):
                polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right = polarities.to(device),pad_target.to(device),att_target.to(device),pad_left.to(device),att_left.to(device),pad_right.to(device),att_right.to(device)
            
                # Forward pass
                output = model(left = pad_left,target = pad_target,right = pad_right,att_left = att_left,att_target = att_target,att_right = att_right)
                i+=1
                #loss = loss_fn(output, torch.argmax(polarities,dim=1))
                #loss.backward()#retain_graph=True)
                #total_loss += loss.item()
                # Update weights
                #optimizer.step()
                
                polarities = torch.argmax(polarities,dim=1)
                pred = torch.argmax(nn.functional.softmax(output,dim=-1),dim=1)
                if batch_idx == 0:
                    y_test = polarities
                    y_pred = pred
                else:
                    y_test = torch.cat((y_test,polarities))
                    y_pred = torch.cat((y_pred,pred))

                polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right = polarities.cpu(),pad_target.cpu(),att_target.cpu(),pad_left.cpu(),att_left.cpu(),pad_right.cpu(),att_right.cpu()
        
    
        neg_indices = torch.nonzero(y_test == 0, as_tuple=True)
        neutral_indices = torch.nonzero(y_test == 1, as_tuple=True)
        pos_indices = torch.nonzero(y_test == 2, as_tuple=True)

        measures = get_measures(y_test=y_test.cpu().numpy(),y_pred=y_pred.cpu().numpy(),samplewise='all')
        neg_measures = get_measures(y_test=y_test[neg_indices].cpu().numpy(),y_pred=y_pred[neg_indices].cpu().numpy())
        neutral_measures = get_measures(y_test=y_test[neutral_indices].cpu().numpy(),y_pred=y_pred[neutral_indices].cpu().numpy())
        pos_measures = get_measures(y_test=y_test[pos_indices].cpu().numpy(),y_pred=y_pred[pos_indices].cpu().numpy())
        print(f'measures: {measures}')
        print(f'neg measures: {neg_measures}')
        print(f'neutral measures: {neutral_measures}')
        print(f'pos measures: {pos_measures}')
    #print(f'Epoch [{epoch+1}/{epochss}], acc: {test_correct / test_total:.4f}, f1: {f1 / len(validation_dataloader):.4f}')
    
    pbar.close()
    
    return measures,neg_measures,neutral_measures,pos_measures

if __name__ == '__main__':

    main()