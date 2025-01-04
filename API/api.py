from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import pickle
import re
import torch
import pandas as pd
from AST import GumTreeAst
import Hierarchical
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import glob
import logging
import subprocess
from utils.llms import ask_llms
from typing import List
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Set up logging to both console and file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Set the default logging level to INFO

# Create a file handler to log to a file
file_handler = logging.FileHandler("app.log")
file_handler.setLevel(logging.INFO)  # Set the level for file logging
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Create a console handler to log to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set the level for console logging
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Lifespan event handler
async def lifespan(app: FastAPI):
    logger.info("Application startup completed.")
    yield
    logger.info("Application shutdown completed.")

# Attach the lifespan event handler to the app
app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (any frontend can make requests)
    allow_credentials=True,  # Allow credentials (cookies, authorization headers, etc.)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (for example, Content-Type, Authorization)
)

# Load model and tokenizer globally to avoid reloading
model_name_or_path = 'MickyMike/VulRepair'
model_path = "model/model_one_stmt_final.bin"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
try:
    tokenizer = RobertaTokenizer.from_pretrained(model_name_or_path)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>", "<S2SV_ModStart>", "<S2SV_ModEnd>"])
except Exception as e:
    raise RuntimeError(f"Error initializing tokenizer: {e}")

# Initialize model
try:
    model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
    model_state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Preload the hierarchical cluster during application startup
hierarchical_cluster = None

def preload_hierarchical_cluster():
    global hierarchical_cluster
    hierarchical_cluster_path = 'database/hierarchical_cluster.pkl'
    
    if os.path.exists(hierarchical_cluster_path):
        logger.info("Loading preprocessed hierarchical cluster...")
        with open(hierarchical_cluster_path, 'rb') as f:
            hierarchical_cluster = pickle.load(f)
    else:
        logger.info("Processing clusters for the first time...")
        hierarchical_cluster = Hierarchical.HierarchicalCluster([])
        clusters_dir = './clusters_3000'
        cluster_paths = [os.path.join(clusters_dir, cluster) for cluster in os.listdir(clusters_dir)]
        
        for cluster_path in cluster_paths:
            with open(cluster_path, 'rb') as f:
                hierarchical_cluster.hierarchical_nodes.extend(pickle.load(f))
        
        with open(hierarchical_cluster_path, 'wb') as f:
            pickle.dump(hierarchical_cluster, f)
        logger.info("Hierarchical cluster saved for future use.")

# Run this function at startup
preload_hierarchical_cluster()

# Helper functions
def convert_to_separator_format(source_code, separator="!@#$"):
    token_pattern = r"""
        [a-zA-Z_][a-zA-Z0-9_]* |  # Match identifiers or keywords
        \d+ |                     # Match integers
        '.*?' |                   # Match single-quoted strings
        \".*?\" |                   # Match double-quoted strings
        [{}()\[\];,] |            # Match single-character operators or delimiters
        [=!<>+\-*/&|^%~]+         # Match multi-character operators
    """
    tokenizer = re.compile(token_pattern, re.VERBOSE)
    tokens = tokenizer.findall(source_code)
    return separator.join(tokens)

def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    return tokens.strip()

@app.post("/process_code/")
async def process_code(
    file: UploadFile = None,
    code: str = Form(None),
    name: str = Form("processed_code")  # Default to "processed_code" if no name is provided
):
    try:
        logger.info("Processing code started.")

        # Read input
        if file:
            logger.info("Received file upload.")
            source_code = (await file.read()).decode("utf-8")
        elif code:
            logger.info("Received code as form input.")
            source_code = code
        else:
            logger.error("No input provided.")
            return JSONResponse(content={"error": "No input provided."}, status_code=400)
        
        # Preprocess the code: remove comments and empty lines
        logger.info("Preprocessing code to remove comments and empty lines.")
        def preprocess_code(code):
            # Remove single-line comments
            code = re.sub(r'//.*', '', code)
            # Remove multi-line comments
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
            # Remove multiple consecutive newlines
            code = re.sub(r'\n\s*\n+', '\n', code)
            # Strip leading and trailing whitespace
            return code.strip()
        
        cleaned_source_code = preprocess_code(source_code)

        logger.info("Step 1: Convert code to separator format.")
        # Step 1: Convert code to separator format
        converted_code = convert_to_separator_format(cleaned_source_code) + '!@#$'
        formatted_code = converted_code.replace(' ', '<S2SV_blank>').replace('!@#$', ' ')

        logger.info("Step 2: Tokenizing input.")
        # Step 2: Tokenize input
        input_ids = tokenizer.encode(
            formatted_code, max_length=512, truncation=True, padding="max_length", return_tensors="pt"
        ).to(device)

        logger.info("Step 3: Performing model inference.")
        # Step 3: Perform model inference   
        with torch.no_grad():
            beam_outputs = model.generate(
                input_ids=input_ids,
                do_sample=False,
                num_beams=8,
                num_return_sequences=4,
                max_length=256,
                diversity_penalty=1.5,
                num_beam_groups=2
            )

        logger.info("Step 4: Decoding predictions.")
        # Step 4: Decode predictions
        predictions = [
            clean_tokens(tokenizer.decode(output, skip_special_tokens=True))
            for output in beam_outputs
        ]

        logger.info("Step 5: Saving results.")
        # Step 5: Save results
        patch_name = name  # Use the provided name
        df = pd.DataFrame({
            'names': [patch_name] * len(predictions),
            'raw_predictions': predictions
        })
        df.index.name = ''
        df.to_csv(f"outputs/raw_predictions.csv", index=True)
        df.to_pickle(f"outputs/predictions.pkl")

        # Modify the filename to add '_nonvul' suffix
        nonvul_filename = f"input/{name}_nonvul.c"

        with open(nonvul_filename, 'w', encoding='utf-8') as f:
            for line in cleaned_source_code.splitlines():
                f.write(line + '\n')

        logger.info("Processing code completed successfully.")
        return {"predictions": predictions}

    except Exception as e:
        logger.error(f"Error during code processing: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

class LLmRequest(BaseModel):
    name: str
    operations: List[str]

@app.post("/process_llm")
async def process_llm(request: LLmRequest):
    file_name = request.name
    operations = request.operations
     # Define the path to the input folder
    input_folder = "input"
    # Build the full filename with prefix and suffix
    file_path = os.path.join(input_folder, f"{file_name}_nonvul.c")

    # Check if the file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found at {file_path}!")
        return JSONResponse(content={"error": "File not found"}, status_code=500)
    
    logger.info(f"Found file at {file_path}!")

    # Read the content of the file
    with open(file_path, "r") as file:
        file_content = file.read()

    # Initialize lists for operations and line numbers
    operation_list = []
    locations = []

    logger.info(f"Start parsing operations")
    # Process operations to extract the actual operation and line numbers
    for operation in operations:
        # Split the operation and the line numbers part using the delimiter
        match = re.match(r"(.+?)(!@#\$)(\[\d+:\d+\])", operation)
        if match:
            action = match.group(1).strip()  # Extract the operation
            line_range = match.group(3).strip('[]').split(':')  # Extract line range
            start_line = int(line_range[0])  # Start line
            end_line = int(line_range[1])    # End line

            action += f' at line {start_line}:{end_line}'

            # Append the operation and line numbers to their respective lists
            operation_list.append(action)
            locations.append({"start_line": start_line, "end_line": end_line})
        else:
            # If the operation doesn't match the expected format, return an error
            logger.error(f"Invalid operation format")
            return JSONResponse(content={"error": "Invalid operation format"}, status_code=400)
        
    logger.info(f"Collecting responses from LLMs")
    response = ask_llms(file_content, operation_list)
    logger.info(response)

    return {"response": response, "operations": operation_list, "locations": locations}

@app.get("/process_pattern")
async def process_pattern(name: str):
    try:
        # Build the command with the provided name parameter
        command = f"python utils/diff.py --name={name}"

        # Execute the command using subprocess
        subprocess.run(command, shell=True, check=True)

        logger.info(f"Running python utils/diff.py --name={name}")

        # Define the output file path
        output_file_path = "outputs/predictions_diff.txt"

        # Ensure the file exists before trying to read it
        if not os.path.exists(output_file_path):
            logger.error(f"Output file {output_file_path} not found.")
            return {"status": "error", "message": f"Output file {output_file_path} not found."}

        logger.info(f"Reading content from {output_file_path}")
        # Read the content of the output file line by line
        with open(output_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        logger.info(f"Clearing content from {output_file_path}")
        # Clear the file after reading it
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.truncate(0)  # Clear the contents of the file

        # Return the lines as a JSON response
        logger.info("Processing pattern completed successfully.")
        return {"status": "success", "output": [line.strip() for line in lines]}

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running the command: {e}")
        return {"status": "error", "message": f"Error running the command: {e}"}
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"status": "error", "message": str(e)}
    
@app.get("/process_ast/")
async def process_ast(name: str):
    try:
        logger.info(f"Processing AST for {name} started.")

        # Check if the '_nonvul.c' file exists
        nonvul_file = f"input/{name}_nonvul.c"
        if not os.path.exists(nonvul_file):
            logger.error(f"Non-vulnerable file '{nonvul_file}' not found.")
            return JSONResponse(content={"error": f"Non-vulnerable file '{nonvul_file}' not found."}, status_code=400)

        # Ensure the hierarchical cluster is loaded
        if hierarchical_cluster is None:
            logger.error("Hierarchical cluster is not loaded.")
            return JSONResponse(content={"error": "Hierarchical cluster is not loaded."}, status_code=500)
        
        logger.info(f"Generating AST using GumTree.")
        # Run gumtree parse command to generate the .c.ast file
        command = f"gumtree parse {nonvul_file} > input/{name}_nonvul.c.ast"
        os.system(command)  # Run the command to generate the AST file

        # Check if the file was generated
        matching_files = glob.glob(f"input/{name}_nonvul.c.ast")
        if not matching_files:
            logger.error(f"Failed to generate AST file for '{name}'.")
            return JSONResponse(content={"error": f"Failed to generate AST file for '{name}'."}, status_code=500)

        # Use the first matched AST file
        ast_file = matching_files[0]
        logger.info(f"Found AST file with the name {ast_file}!")

        # Load the AST
        with open(ast_file) as f:
            lines = f.readlines()
        t = GumTreeAst(lines)

        logger.info(f"Applying patterns with the tree name {name}!")
        # Apply patterns
        rets = hierarchical_cluster.applyPattern(t, name)
        new_asts = rets[1]
        subtree_root_ids = rets[2]
        patterns = rets[0]

        # Prepare the response
        results = []
        seen_codes = set()  # Set to track previously written ast_codes

        for i, pattern in enumerate(patterns):
            ast_codes = [new_ast.getCode() for new_ast in new_asts[i]]
            
            # Process each ast_code
            for ast_code in ast_codes:
                # Skip if the ast_code is a duplicate
                if ast_code in seen_codes:
                    continue
                
                # Add the current ast_code to the seen set
                seen_codes.add(ast_code)
                
                results.append({
                    "pattern_index": i,
                    "pattern_length": len(new_asts[i]),
                })

                # Save unformatted code to a temporary file
                unformatted_file = f"generated/{name}_gen_{i}.c"
                with open(unformatted_file, "w") as f:
                    f.write(ast_code)

                logger.info(f"Processing of AST for {name} completed successfully.")
            
        return {"patterns": results}

    except Exception as e:
        logger.error(f"Error processing AST for {name}: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)