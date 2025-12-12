"""
Calculates Internal Risk Scores and Customer Segmentation Tiers based on customer demographics, 
bureau data, and transaction history. The output report includes risk tiers, review flags, 
and marketing eligibility for each customer.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def import_csv(filename):
    """Utility function to import CSV files into pandas DataFrames."""
    return pd.read_csv(filename)

def calculate_ratios(df):
    """Calculates financial ratios such as debt to income and spend to income."""
    df['estimated_debt'] = (df['num_open_accounts'] * 1500) + (df['delinquencies_2y'] * 500)
    df['dti_ratio'] = df['estimated_debt'] / df['income']
    df['dti_ratio'] = df['dti_ratio'].fillna(0)  # Fix division by zero
    df['sti_ratio'] = df['total_spend'] / df['income']
    df['sti_ratio'] = df['sti_ratio'].fillna(0)  # Fix division by zero
    return df

def score_engine(df):
    """Complex logic to determine Internal Risk Score and Tiers."""
    df['internal_score'] = df['fico_score']
    
    # Rule 1: Income Bonus
    df.loc[df['income'] > 100000, 'internal_score'] += 20
    
    # Rule 2: Delinquency Penalty
    df.loc[df['delinquencies_2y'] > 0, 'internal_score'] -= 50
    
    # Rule 3: DTI Verification
    df.loc[df['dti_ratio'] > 0.4, 'internal_score'] -= 30
    
    # Rule 4: Usage Activity Bonus
    df.loc[df['total_spend'] > 5000, 'internal_score'] += 10
    
    # TIERING LOGIC
    df['risk_tier'] = pd.cut(df['internal_score'], 
                             bins=[-np.inf, 550, 650, 750, np.inf], 
                             labels=['Bronze', 'Silver', 'Gold', 'Platinum'])
    
    # ACTION FLAGS
    df['review_needed'] = np.where((df['risk_tier'] == 'Bronze') | (df['delinquencies_2y'] > 2), 'Y', 'N')
    df['marketing_target'] = np.where((df['risk_tier'] == 'Platinum') & (df['income'] > 80000), 'Y', 'N')
    
    return df

def main():
    input_dir = Path('sas_test_suite/input_data/')
    output_dir = Path('sas_test_suite/generated_outputs/')
    
    # Import datasets
    customers_raw = import_csv(input_dir / 'risk_customers.csv')
    bureau_raw = import_csv(input_dir / 'risk_bureau.csv')
    transactions_raw = import_csv(input_dir / 'risk_transactions.csv')
    
    # Data cleaning and standardization
    customers_raw['state'] = customers_raw['state'].str.upper()
    mean_income = customers_raw['income'].mean()
    customers_raw['income'] = customers_raw['income'].fillna(mean_income)
    customers_raw['income'] = np.maximum(customers_raw['income'], 0)
    
    # Transaction aggregation
    trans_agg = transactions_raw.groupby('cust_id').agg(
        total_spend=('amount', 'sum'), 
        avg_trans_amt=('amount', 'mean'), 
        trans_count=('trans_id', 'count'), 
        max_spend=('amount', 'max')
    ).reset_index()
    
    # Merge data sources
    merged_matrix = pd.merge(customers_raw, bureau_raw, on='cust_id', how='inner')
    merged_matrix = pd.merge(merged_matrix, trans_agg, on='cust_id', how='left')
    merged_matrix = merged_matrix.fillna(0)
    
    # Calculate financial ratios
    data_with_ratios = calculate_ratios(merged_matrix)
    
    # Scoring engine
    scored_data = score_engine(data_with_ratios)
    
    # Final report generation
    final_report = scored_data[['cust_id', 'state', 'income', 'fico_score', 'total_spend', 
                                'dti_ratio', 'internal_score', 'risk_tier', 'review_needed', 
                                'marketing_target']]
    
    # Export to CSV
    final_report.to_csv(output_dir / 'credit_risk_report.csv', index=False)

if __name__ == '__main__':
    main()