import pandas as pd
from scipy import stats


crispr_on = pd.read_excel('../data/main_dataframes/crispr on data.xlsx',engine='openpyxl', sheet_name='spCas9_eff_D10-dox')
crispr_on = crispr_on.rename(columns={'gRNA': 'gRNA_Seq'})

deep_hf = pd.read_csv('../data/main_dataframes/supplementary2.csv')

wt_deep_hf = deep_hf[['gRNA_Seq', 'Wt_Efficiency']]
wt_deep_hf = wt_deep_hf.dropna()

wt_merged = pd.merge(crispr_on, wt_deep_hf, on=['gRNA_Seq'], how='inner')
wt_merged = wt_merged[['gRNA_Seq', 'total_indel_eff', 'Wt_Efficiency']]
wt_merged['total_indel_eff'] = wt_merged['total_indel_eff'].div(100)
spearman = stats.spearmanr(wt_merged['total_indel_eff'], wt_merged['Wt_Efficiency'])[0]
print(spearman)

esp_deep_hf = deep_hf[['gRNA_Seq', 'eSpCas 9_Efficiency']]
esp_deep_hf = esp_deep_hf.dropna()

esp_merged = pd.merge(crispr_on, esp_deep_hf, on=['gRNA_Seq'], how='inner')
esp_merged = esp_merged[['gRNA_Seq', 'total_indel_eff', 'eSpCas 9_Efficiency']]
esp_merged['total_indel_eff'] = esp_merged['total_indel_eff'].div(100)
spearman = stats.spearmanr(esp_merged['total_indel_eff'], esp_merged['eSpCas 9_Efficiency'])[0]
print(spearman)

hf_deep_hf = deep_hf[['gRNA_Seq', 'SpCas9-HF1_Efficiency']]
hf_deep_hf = hf_deep_hf.dropna()

hf_merged = pd.merge(crispr_on, hf_deep_hf, on=['gRNA_Seq'], how='inner')
hf_merged = hf_merged[['gRNA_Seq', 'total_indel_eff', 'SpCas9-HF1_Efficiency']]
hf_merged['total_indel_eff'] = hf_merged['total_indel_eff'].div(100)
spearman = stats.spearmanr(hf_merged['total_indel_eff'], hf_merged['SpCas9-HF1_Efficiency'])[0]
print(spearman)

a=0