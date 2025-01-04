import re
from transformers import T5ForConditionalGeneration, RobertaTokenizer, AutoConfig
import torch
import pandas as pd

def convert_to_separator_format(source_code, separator="!@#$"):
    # Define a regex pattern to match tokens
    token_pattern = r"""
        [a-zA-Z_][a-zA-Z0-9_]* |  # Match identifiers or keywords
        \d+ |                     # Match integers
        '.*?' |                   # Match single-quoted strings
        ".*?" |                   # Match double-quoted strings
        [{}()\[\];,] |            # Match single-character operators or delimiters
        [=!<>+\-*/&|^%~]+         # Match multi-character operators
    """
    # Compile the regex with verbose flag
    tokenizer = re.compile(token_pattern, re.VERBOSE)

    # Find all tokens in the source code
    tokens = tokenizer.findall(source_code)

    # Join tokens with the separator
    return separator.join(tokens)

# Example usage
with open(f'aircrack-ng_CVE-2014-8324_CWE-20_nonvul.c') as f:
    source_code = f.read()

converted_code = convert_to_separator_format(source_code) + '!@#$'
formatted_code = converted_code.replace(' ', '<S2SV_blank>').replace('!@#$', ' ')

tokenizer_name = 'MickyMike/VulRepair'
model_name_or_path = 'MickyMike/VulRepair'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

encoder_block_size = 512
decoder_block_size = 256
num_beams = 10
eval_batch_size = 1

tokenizer = RobertaTokenizer.from_pretrained("MickyMike/VulRepair")
tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])

single_input_ids = tokenizer.encode(formatted_code, max_length=encoder_block_size, truncation=True, padding="max_length", return_tensors="pt")

# Load state dict with CPU mapping
model_path = "model_one_stmt_final.bin"
try:
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    print("Model state dict loaded successfully.")
except Exception as e:
    print(f"Error loading model state dict: {e}")
    exit(1)

# Load model configuration from AutoConfig
try:
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model.load_state_dict(model_state_dict)  # Then load the state dict into the model
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

with torch.no_grad():
    beam_outputs = model.generate(
        input_ids=single_input_ids.to(device),
        do_sample=False,  # Deterministic
        num_beams=8,  # Use 8 beams for top 4 predictions
        num_return_sequences=4,  # Return only the top 4 sequences
        max_length=decoder_block_size,
        diversity_penalty=1.5,
        num_beam_groups=2  # Set number of beam groups (should be > 1 for diversity penalty)
    )

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens

beam_outputs = beam_outputs.detach().cpu().tolist()

predictions = [
    clean_tokens(tokenizer.decode(output, skip_special_tokens=True))
    for output in beam_outputs
]

print(predictions)

patch_name = 'aircrack-ng_CVE-2014-8324_CWE-20'

names = [patch_name] * len(predictions)

df = pd.DataFrame({
    'names': names,
    'raw_predictions': predictions
})

df.index.name = ''

csv_file_path = 'raw_predictions.csv'

df.to_csv(csv_file_path, index=True) 

# Save to .pkl file
pkl_file_path = 'predictions.pkl'
df.to_pickle(pkl_file_path)

print(f"CSV file saved to {csv_file_path}")
print(f"Pickle file saved to {pkl_file_path}")