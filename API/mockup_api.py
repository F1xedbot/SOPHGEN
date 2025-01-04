from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Mock response for /process_code/
@app.post("/process_code/")
async def process_code(
    file: UploadFile = None,
    code: str = Form(None),
    name: str = Form("processed_code")  # Default to "processed_code" if no name is provided
):
    def preprocess_code(code):
        # Remove single-line comments
        code = re.sub(r'//.*', '', code)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove multiple consecutive newlines
        code = re.sub(r'\n\s*\n+', '\n', code)
        # Strip leading and trailing whitespace
        return code.strip()
    
    cleaned_source_code = preprocess_code(code)

    # Modify the filename to add '_nonvul' suffix
    nonvul_filename = f"input/{name}_nonvul.c"

    with open(nonvul_filename, 'w', encoding='utf-8') as f:
        for line in cleaned_source_code.splitlines():
            f.write(line + '\n')

    return JSONResponse(content={
        "predictions": [
            "return match ( prog -> start , sp , sp , prog -> flags | eflags , sub , 0 ) ;",
            "int regexec ( Reprog * prog , const char * sp , Resub * sub , int eflags ) return match ( prog -> start , sp , sp , prog -> flags | eflags , sub , 0 ) ;",
            "int regexec ( Reprog * prog , const char * sp , Resub * sub , int eflags ) sub -> nsub = prog -> nsub ;",
            "int regexec ( Reprog * prog , const char * sp , Resub * sub , int eflags ) if ( ! sub ) sub = & scratch ;"
        ]
    })

# Pydantic model for /process_llm/
class LLmRequest(BaseModel):
    name: str
    operations: List[str]

# Mock response for /process_llm/
@app.post("/process_llm")
async def process_llm(request: LLmRequest):
    return JSONResponse(content={
        "response": "1. `DELETE: if (!sub) sub = &scratch;`\nVulnerable: Null Pointer Dereference, Reason: Removing the null check could cause a segmentation fault if a null pointer is passed, potentially leading to program crash or undefined behavior.\n\n2. `UPDATE: int i; -> int i = 0;`\nAlternative: `int i = -1;`, Reason: Initializing with -1 could potentially cause off-by-one errors or unexpected array indexing behavior, Vulnerable: Initialization Vulnerability",
        "operations": [
            "DELETE: if (!sub) sub = &scratch;",
            "UPDATE: int i; -> int i = 0;"
        ],
        "locations": [
            {
                "start_line": 5,
                "end_line": 6
            },
            {
                "start_line": 9,
                "end_line": 9
            }
        ]
    })

# Mock response for /process_pattern/
@app.get("/process_pattern")
async def process_pattern(name: str):
    return JSONResponse(content={
        "status": "success",
        "output": [
            "DELETE: if (!sub) sub = &scratch;!@#$[5:6]"
        ]
    })

# Mock response for /process_ast/
@app.get("/process_ast/")
async def process_ast(name: str):
    return JSONResponse(content={
        "patterns": [
            {
                "pattern_index": 0,
                "pattern_length": 1
            }
        ]
    })
