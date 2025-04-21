import pandas as pd
import math

# Load the re-uploaded CSV file
file_path = "OE UD Cairo - Sheet1.csv"
df = pd.read_csv(file_path)

# Show the first few rows to verify the structure
df.head()

# Prepare for CoNLL-U conversion
from collections import defaultdict

# Clean the DataFrame
df = df.dropna(how='all', axis=1)  # Drop completely empty columns
df['SentID'] = df['SentID'].ffill()  # Forward fill sentence IDs for grouping

# CoNLL-U columns
CONLLU_COLUMNS = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]

# Morphological features to compile into FEATS
morph_feats = ["Case", "Number", "Gender", "Tense", "Person", "VerbForm", "Mood", "ExtPos"]

def int_wrapper(a):
    if type(a) == float:
        return int(a)
    else:
        return a

# Function to compile feats
def compile_feats(row):
    feats = [f"{feat}={int_wrapper(row[feat])}" for feat in morph_feats if pd.notna(row.get(feat))]
    return "|".join(sorted(feats)) if feats else "_"

# Function to compile misc
def compile_misc(row):
    misc_items = []
    for col in ['Gloss', 'Root', 'Hyperlemma', 'Origlang']:
        if col in row and pd.notna(row[col]):
            misc_items.append(f"{col}={row[col]}")
    return "|".join(sorted(misc_items)) if misc_items else "_"

# Group by sentence and format into CoNLL-U
conllu_output = []
for sent_id, sent_df in df.groupby('SentID'):
    sent = []
    local_output = []
    for _, row in sent_df.iterrows():
        if math.isnan(row["ID"]):
            continue
        feats = compile_feats(row)
        misc = compile_misc(row)
        conllu_row = [
            str(int(row["ID"])) if pd.notna(row["ID"]) else "_",
            row["FORM"] if pd.notna(row["FORM"]) else "_",
            row["LEMMA"] if pd.notna(row["LEMMA"]) else "_",
            row["UPOS"] if pd.notna(row["UPOS"]) else "_",
            row["XPOS"] if pd.notna(row["XPOS"]) else "_",
            feats,
            str(int(row["HEAD"])) if pd.notna(row["HEAD"]) else "_",
            row["DEPREL"] if pd.notna(row["DEPREL"]) else "_",
            "_",  # DEPS
            misc
        ]
        local_output.append("\t".join(conllu_row))
        sent.append(row["FORM"])
    conllu_output.append(f"# sent_id = {int(sent_id)}")
    conllu_output.append(f"# text = {' '.join(sent)}")
    conllu_output.extend(local_output)
    conllu_output.append("")  # Empty line between sentences

# Save to file
conllu_path = "output.conllu"
with open(conllu_path, "w", encoding="utf-8") as f:
    f.write("\n".join(conllu_output))
