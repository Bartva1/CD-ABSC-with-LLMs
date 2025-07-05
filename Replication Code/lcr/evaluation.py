"""
This file is adapted from:
"Domain-Adaptive Aspect-Based Sentiment Classification Using Masked Domain-Specific Words and Target Position-Aware Attention"
by Finn van der Knaap (https://github.com/FvdKnaap/DAWM-LCR-Rot-hop-plus-plus).

Provides evaluation utilities for LCR Rot-hop++ models, including accuracy, precision, recall, and F1 metrics.
"""

from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score

def get_measures(y_test,y_pred,samplewise = 'per'):
    if samplewise == 'all':
        acc = accuracy_score(y_true=y_test,y_pred=y_pred)
        precision = precision_score(y_true=y_test,y_pred=y_pred,average='macro')
        recall = recall_score(y_true=y_test,y_pred=y_pred,average='macro')
        f1 = f1_score(y_true=y_test,y_pred=y_pred,average='macro')
        
        micro_precision = precision_score(y_true=y_test,y_pred=y_pred,average='micro')
        micro_recall = recall_score(y_true=y_test,y_pred=y_pred,average='micro')
        micro_f1 = f1_score(y_true=y_test,y_pred=y_pred,average='micro')


        return {'acc': acc,'precision': precision,'recall': recall, 'f1': f1, 'micro_precision': micro_precision,'micro_recall': micro_recall, 'micro_f1': micro_f1}
    else:
        
        acc = accuracy_score(y_true=y_test,y_pred=y_pred)
        precision = precision_score(y_true=y_test,y_pred=y_pred,average='micro')
        recall = recall_score(y_true=y_test,y_pred=y_pred,average='micro')
        f1 = f1_score(y_true=y_test,y_pred=y_pred,average='micro')
    
        return {'acc': acc,'precision': precision,'recall': recall, 'f1': f1}