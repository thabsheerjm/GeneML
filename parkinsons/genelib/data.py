import numpy as np
import pandas as pd

def log2_transform(data):
    if data.max().max() > 100:
        data_transformed = np.log2(data+1)
    return data_transformed

def get_top_genes_linear(model, feature_names, top_n=10):
    coeffs = pd.DataFrame({'Gene': feature_names, 'Weight':model.coef_[0]})
    coeffs['Abs_Weight'] = coeffs['Weight'].abs()

    top_genes = coeffs.sort_values('Abs_Weight', ascending=False).head(top_n)
    return top_genes[['Gene', 'Weight']].reset_index(drop=True)

def get_top_genes_rf(model, feature_names, top_n=10):
    importance = pd.DataFrame({'Gene': feature_names, 'Importance': model.feature_importances_})
    top_genes = importance.sort_values('Importance', ascending=False).head(top_n)
    return top_genes[['Gene', 'Importance']].reset_index(drop=True)

def create_consensus_table(df_ttest, log_model, svm_model, rf_model, selected_genes, top_n=10):
    '''
    Creates a consensus table comparing ttest p-values against ML model weights.
    '''
    df_t_sorted = df_ttest.sort_values('p_value').head(top_n)
    df_t_sorted.index.name = 'Gene'
    df_t_sorted = df_t_sorted.reset_index()

    df_log = get_top_genes_linear(log_model, selected_genes, top_n=len(selected_genes))
    df_svm = get_top_genes_linear(svm_model, selected_genes, top_n=len(selected_genes))
    df_rf  = get_top_genes_rf(rf_model, selected_genes, top_n=len(selected_genes))

    #rename and  merge
    df_log = df_log.rename(columns={'Weight': 'LogReg_Weight'})
    df_svm = df_svm.rename(columns={'Weight': 'SVM_Weight'})
    df_rf  = df_rf.rename(columns={'Importance': 'RF_Imp'})

    merged = pd.merge(df_t_sorted, df_log, on='Gene', how='left')
    merged = pd.merge(merged, df_svm, on='Gene', how='left')
    merged = pd.merge(merged, df_rf, on='Gene', how='left')

    df_rf['RF_Rank_Global'] = df_rf['RF_Imp'].rank(ascending=False)
    rank_map = dict(zip(df_rf['Gene'], df_rf['RF_Rank_Global']))
    merged['RF_Rank'] = merged['Gene'].map(rank_map)

    merged['Strong Consensus'] = merged['RF_Rank'] <= top_n
    final_cols = ['Gene', 'p_value', 'LogReg_Weight', 'SVM_Weight', 'RF_Imp', 'RF_Rank', 'Strong Consensus']

    if merged.isnull().values.any():
        print("WARNING!!: Some T-Test genes were not found in the Model Features.")
        print("Check for name mismatches")
    return merged[final_cols]
