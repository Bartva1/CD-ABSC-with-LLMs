import torch
from load_data import CustomDataset,CustomDataset2
from torch.utils.data import DataLoader
from config import *
from bertmasker_lcr import SentimentClassifier, SharedPart, BERTMasker_plus
import torch.nn as nn
from tqdm import tqdm
from lcr_rot_hopplusplus import LCRRotHopPlusPlus
import torchmetrics
from evaluation import get_measures
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import os
import numpy as np
import glob

def split_indices(dataset_len, n_splits=3):
    indices = np.arange(dataset_len)
    np.random.shuffle(indices)
    return np.array_split(indices, n_splits)

def load_train(domain,target_values):
    """
    Load training data for a specified domain.
    
    Args:
        domain (str): The domain for which to load the training data.
        target_values (int): The domains used for 
    
    Returns:
        tuple: A tuple containing the training dataset, mask embeddings, and pad embeddings.
    """
    out_path = f'train/variables_{domain}/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt', weights_only=True)
    token_ids = torch.load(out_path + 'token_ids.pt', weights_only=True)
    segment_ids = torch.load(out_path + 'segment_ids.pt', weights_only=True)
    polarities = torch.load(out_path + 'polarities.pt', weights_only=True)
    domain = torch.load(out_path + 'domain.pt', weights_only=True)
    target_ind = torch.load(out_path + 'target_ind.pt', weights_only=True)
    masking_constraints = torch.load(out_path + 'masking_constraints.pt', weights_only=True)
    input_embeddings = torch.load(out_path + 'input_embeddings.pt', weights_only=True)
    domain_list = torch.load(out_path + 'domain_list.pt', weights_only=True)

    pad_target = torch.load(out_path + 'pad_target.pt', weights_only=True)
    att_target = torch.load(out_path + 'att_target.pt', weights_only=True)
    pad_left = torch.load(out_path + 'pad_left.pt', weights_only=True)
    att_left = torch.load(out_path + 'att_left.pt', weights_only=True)
    pad_right = torch.load(out_path + 'pad_right.pt', weights_only=True)
    att_right = torch.load(out_path + 'att_right.pt', weights_only=True)

    mask_embedding = torch.load(out_path + 'mask_embedding.pt', weights_only=True)
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt', weights_only=True)

    return CustomDataset2(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right,target_tensor_index=8,target_values=target_values),mask_embedding,pad_embedding

def load_test(domain):
    out_path = f'test/variables{domain}/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt', weights_only=True)
    token_ids = torch.load(out_path + 'token_ids.pt', weights_only=True)
    segment_ids = torch.load(out_path + 'segment_ids.pt', weights_only=True)
    polarities = torch.load(out_path + 'polarities.pt', weights_only=True)
    domain = torch.load(out_path + 'domain.pt', weights_only=True)
    target_ind = torch.load(out_path + 'target_ind.pt', weights_only=True)
    masking_constraints = torch.load(out_path + 'masking_constraints.pt', weights_only=True)
    input_embeddings = torch.load(out_path + 'input_embeddings.pt', weights_only=True)
    domain_list = torch.load(out_path + 'domain_list.pt', weights_only=True) 

    pad_target = torch.load(out_path + 'pad_target.pt', weights_only=True)
    att_target = torch.load(out_path + 'att_target.pt', weights_only=True)
    pad_left = torch.load(out_path + 'pad_left.pt', weights_only=True)
    att_left = torch.load(out_path + 'att_left.pt', weights_only=True)
    pad_right = torch.load(out_path + 'pad_right.pt', weights_only=True)
    att_right = torch.load(out_path + 'att_right.pt', weights_only=True)

    mask_embedding = torch.load(out_path + 'mask_embedding.pt', weights_only=True)
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt', weights_only=True)

    return CustomDataset(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right)

def load_train2(domain, chunk_idx=0, chunk_dir="chunked"):
    out_path = f"train/variables_{domain}/{chunk_dir}/"
    token_embeddings = torch.load(os.path.join(out_path, f'token_embeddings_{chunk_idx}.pt'))
    token_ids = torch.load(os.path.join(out_path, f'token_ids_{chunk_idx}.pt'))
    segment_ids = torch.load(os.path.join(out_path, f'segment_ids_{chunk_idx}.pt'))
    polarities = torch.load(os.path.join(out_path, f'polarities_{chunk_idx}.pt'))
    domain_tensor = torch.load(os.path.join(out_path, f'domain_{chunk_idx}.pt'))
    target_ind = torch.load(os.path.join(out_path, f'target_ind_{chunk_idx}.pt'))
    masking_constraints = torch.load(os.path.join(out_path, f'masking_constraints_{chunk_idx}.pt'))
    input_embeddings = torch.load(os.path.join(out_path, f'input_embeddings_{chunk_idx}.pt'))
    domain_list = torch.load(os.path.join(out_path, f'domain_list_{chunk_idx}.pt'))
    pad_target = torch.load(os.path.join(out_path, f'pad_target_{chunk_idx}.pt'))
    att_target = torch.load(os.path.join(out_path, f'att_target_{chunk_idx}.pt'))
    pad_left = torch.load(os.path.join(out_path, f'pad_left_{chunk_idx}.pt'))
    att_left = torch.load(os.path.join(out_path, f'att_left_{chunk_idx}.pt'))
    pad_right = torch.load(os.path.join(out_path, f'pad_right_{chunk_idx}.pt'))
    att_right = torch.load(os.path.join(out_path, f'att_right_{chunk_idx}.pt'))
    mask_embedding = torch.load(os.path.join(out_path, 'mask_embedding.pt'))
    pad_embedding  = torch.load(os.path.join(out_path, 'pad_embedding.pt'))

    return CustomDataset(token_embeddings,token_ids,segment_ids,polarities,domain_tensor,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right), mask_embedding, pad_embedding

def get_hyperparameters(source_domain, target_domain):
  
    hyperparams = {
        ('restaurant', 'laptop'): {
            'lr': 0.0005,
            'weight_decay': 0.001,
            'hidden_s': 256,
            'temp': 0.01,
            'alp': 1,
            'weight_shared': 0.005,
            'weight_private': 0.005,
            'weight_sent': 4
        },
        ('restaurant', 'book'): {
            'lr': 0.0001,
            'weight_decay': 0.001,
            'hidden_s': 256,
            'temp': 0.01,
            'alp': 0.5,
            'weight_shared': 0.005,
            'weight_private': 0.005,
            'weight_sent': 3
        },
        ('laptop', 'restaurant'): {
            'lr': 0.0005,
            'weight_decay': 0.001,
            'hidden_s': 64,
            'temp': 0.01,
            'alp': 1.5,
            'weight_shared': 0.005,
            'weight_private': 0.005,
            'weight_sent': 3
        },
        ('laptop', 'book'): {
            'lr': 0.0001,
            'weight_decay': 0.001,
            'hidden_s': 256,
            'temp': 0.01,
            'alp': 0.5,
            'weight_shared': 0.005,
            'weight_private': 0.005,
            'weight_sent': 3
        },
        ('book', 'restaurant'): {
            'lr': 0.01,
            'weight_decay': 0.005,
            'hidden_s': 64,
            'temp': 0.1,
            'alp': 2,
            'weight_shared': 0.01,
            'weight_private': 0.01,
            'weight_sent': 1
        },
        ('book', 'laptop'): {
            'lr': 0.01,
            'weight_decay': 0.01,
            'hidden_s': 128,
            'temp': 0.5,
            'alp': 1.5,
            'weight_shared': 0.005,
            'weight_private': 0.005,
            'weight_sent': 4
        }
    }
    
    params = hyperparams.get((source_domain, target_domain))
    if params is None:
        raise ValueError(f"No hyperparameters found for source-target pair: {source_domain}-{target_domain}")
    return params


def evaluate_saved_predictions(result_dir, train_domain, test_domain):
    result_path = os.path.join(result_dir, f"{train_domain}_{test_domain}.pt")
    if not os.path.exists(result_path):
        print(f"No prediction file found at: {result_path}")
        return None

    saved = torch.load(result_path, map_location='cpu')
    y_true = saved['y_true'].numpy()
    y_pred = saved['y_pred'].numpy()

    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Per-class accuracy
    classes = np.unique(y_true)
    per_class_acc = {}
    for c in classes:
        idx = (y_true == c)
        per_class_acc[int(c)] = accuracy_score(y_true[idx], y_pred[idx]) if np.sum(idx) > 0 else float('nan')

    output_dir = os.path.join(result_dir, "metrics/")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{train_domain}_{test_domain}.txt")
    with open(output_path, 'w') as f:
        f.write(f"Accuracy: {100 * acc:.2f} \n")
        f.write(f"Macro-Precision: {100 * macro_precision:.2f} \n")
        f.write(f"Macro-Recall: {100 *macro_recall:.2f} \n")
        f.write(f"Macro-F1: {100 *macro_f1:.2f} \n")
        f.write("Per-class accuracy: \n")
        for c, a in per_class_acc.items():
            f.write(f"  Class {c}: {100 * a:.2f} \n")

    return acc, macro_precision, macro_recall, macro_f1, per_class_acc


def main(source_domain, target_domain):
    result_dir = "results/dawm_lcr_rot_hop"
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f"{source_domain}_{target_domain}.pt")
    if os.path.exists(result_path):
        print(f"Predictions for {source_domain} → {target_domain} already exist at {result_path}. Skipping.")
        return

    domain_map = {"laptop": 0, "restaurant": 1, "book": 2}
    if source_domain not in domain_map or target_domain not in domain_map:
        raise ValueError("Invalid domain name. Choose from: laptop, restaurant, book.")

    source_idx = domain_map[source_domain]
    target_idx = domain_map[target_domain]
    targets = [source_idx]

    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Find number of chunks
    chunk_dir = f"train/variables_{source_domain}/chunked/"
    chunk_files = sorted(glob.glob(os.path.join(chunk_dir, 'token_embeddings_*.pt')))
    n_chunks = len(chunk_files)
    print(f"Found {n_chunks} training chunks.")

    # Load mask and pad embedding (small, so just load from chunk 0)
    _, mask_embedding, pad_embedding = load_train2(source_domain, chunk_idx=0)
    mask_embedding = mask_embedding.to(device)
    pad_embedding = pad_embedding.to(device)

    params = get_hyperparameters(source_domain, target_domain)
    hidden_s = params['hidden_s']
    lr = params['lr']
    weight_decay = params['weight_decay']
    epochs = 8
    weight_shared = params['weight_shared']
    weight_private = params['weight_private']
    temp = params['temp']
    weight_sent = params['weight_sent']
    alp = params['alp']

    shared_part = SharedPart(hidden_size=hidden_s, temp=temp, alpha=alp, masking=0.1).to(device)
    private_part = None
    sentiment_classifier = SentimentClassifier().to(device)
    shared_lcr = LCRRotHopPlusPlus(sentiment_prediction=False).to(device)
    private_lcr = LCRRotHopPlusPlus(sentiment_prediction=False).to(device)
    model = BERTMasker_plus(
        shared_domain_classifier=shared_part,
        private_domain_classifier=private_part,
        shared_lcr=shared_lcr,
        private_lcr=private_lcr,
        sentiment_classifier=sentiment_classifier
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    shared_loss_fn = nn.CrossEntropyLoss()
    private_loss_fn = nn.CrossEntropyLoss()
    sentiment_loss_fn = nn.CrossEntropyLoss()

    train_acc_prev = torch.tensor(0.0, device=device)
    train_acc_prev2 = torch.tensor(0.0, device=device)
    train_acc_prev3 = torch.tensor(0.0, device=device)

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_shared = 0.0
        total_private = 0.0
        total_sentiment = 0.0
        train_correct = 0
        train_total = 0.0

        # Loop over all chunks for this epoch
        for chunk_idx in range(n_chunks):
            dataset, _, _ = load_train2(source_domain, chunk_idx=chunk_idx)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Set requires_grad for different epochs
            if epoch < 1:
                for name, param in model.named_parameters():
                    if 'shared_lcr' in name or 'private_lcr' in name or 'sentiment_classifier' in name:
                        param.requires_grad = False
            elif epoch < 10:
                for name, param in model.named_parameters():
                    if 'shared_lcr' in name or 'private_lcr' in name or 'sentiment_classifier' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            else:
                for name, param in model.named_parameters():
                    param.requires_grad = True

            with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochs} Chunk {chunk_idx+1}/{n_chunks}", unit="batch") as pbar:
                for batch_idx, (hidden_embeddings, _, segments_tensor, polarities, domain, target_indices, _, input_embedding, domain_list, _, _, _, _, _, _) in enumerate(data_loader):
                    if epoch > 0:
                        domain_list = torch.zeros(domain_list.size(), device=device, dtype=torch.int64)
                    optimizer.zero_grad()
                    hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding, target_indices = \
                        hidden_embeddings.to(device), segments_tensor.to(device), polarities.to(device), domain.to(device), domain_list.to(device), input_embedding.to(device), target_indices.to(device)

                    shared_output, private_output, sentiment_pred, mask_perc, input_em = model(
                        hidden_embeddings=hidden_embeddings,
                        input_embedding=input_embedding,
                        mask_embedding=mask_embedding,
                        pad_embedding=pad_embedding,
                        segments_tensor=segments_tensor,
                        domain_list=domain_list,
                        target_ind=target_indices
                    )

                    if epoch < 1:
                        shared_loss = shared_loss_fn(shared_output, domain_list)
                        private_loss = private_loss_fn(private_output, domain_list)
                        epoch_loss = weight_shared * shared_loss + weight_private * private_loss
                        total_loss += epoch_loss.item()
                        total_shared += shared_loss.item()
                        total_private += private_loss.item()
                    elif epoch < 10:
                        sentiment_loss = sentiment_loss_fn(sentiment_pred, torch.argmax(polarities, dim=1))
                        epoch_loss = weight_sent * sentiment_loss
                        total_loss += epoch_loss.item()
                    else:
                        shared_loss = shared_loss_fn(shared_output, domain_list)
                        private_loss = private_loss_fn(private_output, domain_list)
                        sentiment_loss = sentiment_loss_fn(sentiment_pred, torch.argmax(polarities, dim=1))
                        epoch_loss = weight_shared * shared_loss + weight_private * private_loss + weight_sent * sentiment_loss
                        total_loss += epoch_loss.item()
                        total_shared += shared_loss.item()
                        total_private += private_loss.item()
                        total_sentiment += sentiment_loss.item()

                    train_correct += torch.sum(
                        (torch.argmax(nn.functional.softmax(sentiment_pred, dim=-1), dim=1) == torch.argmax(polarities, dim=1)).type(torch.int)
                    ).item()
                    train_total += polarities.size(0)

                    epoch_loss.backward(retain_graph=True)
                    optimizer.step()

                    # Move tensors back to CPU to free GPU memory
                    hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding, target_indices = \
                        hidden_embeddings.cpu(), segments_tensor.cpu(), polarities.cpu(), domain.cpu(), domain_list.cpu(), input_embedding.cpu(), target_indices.cpu()

                    pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                    pbar.update(1)

        print(f'Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss:.4f}, shared loss {total_shared:.4f}, private loss: {total_private:.4f}, sentiment loss: {total_sentiment:.4f}')
        train_acc = torch.tensor(train_correct / train_total, device=device)
        print(f'acc: {train_acc}')
        train_acc_prev3 = train_acc_prev2
        train_acc_prev2 = train_acc_prev
        train_acc_prev = train_acc

        # Early stopping logic can go here if desired

    # --- Evaluation on test set (not chunked) ---
    domain = f"_{target_domain}"
    test_dataset = load_test(domain)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    sentence_list = []
    with torch.no_grad():
        for batch_idx, (hidden_embeddings, _, segments_tensor, polarities, domain, target_indices, _, input_embedding, domain_list, _, _, _, _, _, _) in enumerate(test_dataloader):
            hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = \
                hidden_embeddings.to(device), segments_tensor.to(device), polarities.to(device), domain.to(device), domain_list.to(device), input_embedding.to(device)

            domain_list = torch.ones(domain_list.size(), device=device, dtype=torch.int64)
            shared_output, private_output, sentiment_pred, mask_perc, sentence_emb = model(
                hidden_embeddings=hidden_embeddings,
                input_embedding=input_embedding,
                mask_embedding=mask_embedding,
                pad_embedding=pad_embedding,
                segments_tensor=segments_tensor,
                domain_list=domain_list,
                target_ind=target_indices
            )

            polarities = torch.argmax(polarities, dim=1)
            pred = torch.argmax(nn.functional.softmax(sentiment_pred, dim=-1), dim=1)
            shared_pred = torch.argmax(nn.functional.softmax(shared_output, dim=-1), dim=1)
            private_pred = torch.argmax(nn.functional.softmax(private_output, dim=-1), dim=1)

            if batch_idx == 0:
                y_test = polarities
                y_pred = pred
                domain_test = domain_list
                domain_pred_shared = shared_pred
                domain_pred_private = private_pred
                total_mask = mask_perc
                sentence_list.append(sentence_emb)
            else:
                y_test = torch.cat((y_test, polarities))
                y_pred = torch.cat((y_pred, pred))
                domain_test = torch.cat((domain_test, domain_list))
                domain_pred_shared = torch.cat((domain_pred_shared, shared_pred))
                domain_pred_private = torch.cat((domain_pred_private, private_pred))
                total_mask = torch.cat((total_mask, mask_perc))
                sentence_list.append(sentence_emb)

            hidden_embeddings, segments_tensor, polarities, domain, domain_list, input_embedding = \
                hidden_embeddings.cpu(), segments_tensor.cpu(), polarities.cpu(), domain.cpu(), domain_list.cpu(), input_embedding.cpu()

    shared_measures = get_measures(y_test=domain_test.cpu().numpy(), y_pred=domain_pred_shared.cpu().numpy())
    private_measures = get_measures(y_test=domain_test.cpu().numpy(), y_pred=domain_pred_private.cpu().numpy())

    print(f'shared measures: {shared_measures}')
    print(f'private measures: {private_measures}')

    total_mask = total_mask.cpu().numpy()
    # Optionally plot or save results here

    # Save predictions and true labels
    torch.save({
        'y_true': y_test.cpu(),
        'y_pred': y_pred.cpu(),
        'domain_true': domain_test.cpu(),
        'domain_pred_shared': domain_pred_shared.cpu(),
        'domain_pred_private': domain_pred_private.cpu(),
        'mask_perc': total_mask,
    }, result_path)
    print(f"Saved predictions to {result_path}")

if __name__ == '__main__':
   
 
    train_domains = ["laptop","book", "restaurant"]
    test_domains = ["laptop", "book", "restaurant"]

    for train_domain in train_domains:
        for test_domain in test_domains:
            if train_domain == test_domain: 
                continue
            # result_dir = "results/lcr_rot_hop/"
            # evaluate_saved_predictions(result_dir, train_domain, test_domain)
            # continue
            main(train_domain, test_domain)
            result_dir = "results/dawm_lcr_rot_hop/"
            print(f"\n=== Evaluation for {train_domain} → {test_domain} ===")
            evaluate_saved_predictions(result_dir, train_domain, test_domain)