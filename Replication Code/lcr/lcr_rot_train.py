import torch
from load_data import CustomDataset
from torch.utils.data import DataLoader
from config import *
from lcr_rot_hopplusplus import LCRRotHopPlusPlus
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
from evaluation import get_measures
import os

def load_train(domain):
    out_path = f"train/variables_{domain}/"
    #token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    #token_ids = torch.load(out_path + 'token_ids.pt')
    #segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt', weights_only = True)
    # domain = torch.load(out_path + 'domain.pt', weights_only = True)
    #target_ind = torch.load(out_path + 'target_ind.pt')
    #masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    #input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    #domain_list = torch.load(out_path + 'domain_list.pt')
    
    pad_target = torch.load(out_path + 'pad_target.pt', weights_only = True)
    att_target = torch.load(out_path + 'att_target.pt', weights_only = True)
    pad_left = torch.load(out_path + 'pad_left.pt',  weights_only = True)
    att_left = torch.load(out_path + 'att_left.pt',  weights_only = True)
    pad_right = torch.load(out_path + 'pad_right.pt', weights_only = True)
    att_right = torch.load(out_path + 'att_right.pt', weights_only = True)

    #mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    #pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right)

def load_test(domain):
    out_path = 'test/variables_' + domain + '/'
    #token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    #token_ids = torch.load(out_path + 'token_ids.pt')
    #segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt',weights_only=True)
    domain = torch.load(out_path + 'domain.pt', weights_only=True)
    #target_ind = torch.load(out_path + 'target_ind.pt')
    #masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    #input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    #domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt', weights_only = True)
    att_target = torch.load(out_path + 'att_target.pt', weights_only = True)
    pad_left = torch.load(out_path + 'pad_left.pt',  weights_only = True)
    att_left = torch.load(out_path + 'att_left.pt',  weights_only = True)
    pad_right = torch.load(out_path + 'pad_right.pt', weights_only = True)
    att_right = torch.load(out_path + 'att_right.pt', weights_only = True)

    #mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    #pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(polarities,pad_target,att_target,pad_left,att_left,pad_right,att_right)


def main(train_domain, test_domain):
    dataset = load_train(train_domain)
    test_data = load_test(test_domain)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    
    param = {'lr': 0.0005, 'weight_decay': 0.01}
    batch_size = 1
    print('loaded data')
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    result_dir = f"results/lcr_rot_hop"
    os.makedirs(result_dir, exist_ok=True)
    model_path = os.path.join(result_dir, f"model_{train_domain}.pt")

    model = LCRRotHopPlusPlus().to(device)
    result_path = os.path.join(result_dir, f"{train_domain}_{test_domain}.pt")
    if os.path.exists(result_path):
        print(f"Predictions already exist for {train_domain} → {test_domain}, skipping.")
        return
    if os.path.exists(model_path):
        print(f"Model found at {model_path}, loading instead of training.")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=param['lr'], weight_decay=param['weight_decay'])
        loss_fn = nn.CrossEntropyLoss()
        num_epochs = 7
        i = 0
        train_acc_prev = torch.tensor(0.0, device=device)
        train_acc_prev2 = torch.tensor(0.0, device=device)
        train_acc_prev3 = torch.tensor(0.0, device=device)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            train_acc = 0.0
            train_correct = 0
            train_total = 0.0

            with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
                for batch_idx, (polarities, pad_target, att_target, pad_left, att_left, pad_right, att_right) in enumerate(data_loader):
                    optimizer.zero_grad()
                    polarities, pad_target, att_target, pad_left, att_left, pad_right, att_right = \
                        polarities.to(device), pad_target.to(device), att_target.to(device), \
                        pad_left.to(device), att_left.to(device), pad_right.to(device), att_right.to(device)

                    output = model(left=pad_left, target=pad_target, right=pad_right,
                                   att_left=att_left, att_target=att_target, att_right=att_right)
                    i += 1
                    loss = loss_fn(output, torch.argmax(polarities, dim=1))
                    train_acc += torchmetrics.functional.accuracy(torch.argmax(nn.functional.softmax(output, dim=-1), dim=1),
                                                                  torch.argmax(polarities, dim=1),
                                                                  task='multiclass', num_classes=num_polarities)
                    train_total += polarities.size(0)
                    train_correct += torch.sum((torch.argmax(nn.functional.softmax(output, dim=-1), dim=1) ==
                                                torch.argmax(polarities, dim=1)).int()).item()
                    loss.backward()
                    total_loss += loss.item()
                    optimizer.step()

                    pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                    pbar.update(1)

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, acc: {train_acc / len(data_loader):.4f}, correct: {train_correct}')
            train_acc = torch.tensor(train_correct / train_total, device=device)
            train_acc_prev3 = train_acc_prev2
            train_acc_prev2 = train_acc_prev
            train_acc_prev = train_acc

            if torch.max(train_acc_prev2,train_acc_prev) - train_acc_prev3 < eps:
                break

        # Save trained model
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

    # === Evaluation (runs in both cases) ===
    model.eval()
    with torch.no_grad():
        for batch_idx, (polarities, pad_target, att_target, pad_left, att_left, pad_right, att_right) in enumerate(test_data_loader):
            polarities, pad_target, att_target, pad_left, att_left, pad_right, att_right = \
                polarities.to(device), pad_target.to(device), att_target.to(device), \
                pad_left.to(device), att_left.to(device), pad_right.to(device), att_right.to(device)

            output = model(left=pad_left, target=pad_target, right=pad_right,
                           att_left=att_left, att_target=att_target, att_right=att_right)
            pred = torch.argmax(nn.functional.softmax(output, dim=-1), dim=1)
            true = torch.argmax(polarities, dim=1)

            if batch_idx == 0:
                y_test = true
                y_pred = pred
            else:
                y_test = torch.cat((y_test, true))
                y_pred = torch.cat((y_pred, pred))

    
    torch.save({'y_true': y_test.cpu(), 'y_pred': y_pred.cpu()}, result_path)
    print(f"Saved predictions to {result_path}")

    

def evaluate_saved_predictions(result_path):
    """Load saved predictions and compute evaluation metrics."""

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"No prediction file found at: {result_path}")

    saved = torch.load(result_path, weights_only = True)
    y_true = saved['y_true']
    y_pred = saved['y_pred']

    neg_indices = torch.nonzero(y_true == 0, as_tuple=True)
    neutral_indices = torch.nonzero(y_true == 1, as_tuple=True)
    pos_indices = torch.nonzero(y_true == 2, as_tuple=True)

    measures = get_measures(y_test=y_true.numpy(), y_pred=y_pred.numpy(), samplewise='all')
    neg_measures = get_measures(y_test=y_true[neg_indices].numpy(), y_pred=y_pred[neg_indices].numpy())
    neutral_measures = get_measures(y_test=y_true[neutral_indices].numpy(), y_pred=y_pred[neutral_indices].numpy())
    pos_measures = get_measures(y_test=y_true[pos_indices].numpy(), y_pred=y_pred[pos_indices].numpy())

    return measures, neg_measures, neutral_measures, pos_measures

if __name__ == '__main__':
    train_domains = ["restaurant","laptop", "book"]
    test_domains = ["restaurant", "laptop", "book"]

    for train_domain in train_domains:
        for test_domain in test_domains:
            if train_domain == test_domain: 
                continue
            main(train_domain, test_domain)
            result_dir = "results/lcr_rot_hop"
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, f"{train_domain}_{test_domain}.pt")
            overall, negative, neutral, positive = evaluate_saved_predictions(result_path)
            print("=== Overall Performance ===")
            print(overall)

            # print("\n--- Class-wise Performance ---")
            # print("Negative:", negative)
            # print("Neutral:", neutral)
            # print("Positive:", positive)

    