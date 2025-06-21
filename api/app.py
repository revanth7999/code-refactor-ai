from fastapi import FastAPI, Request
from pydantic import BaseModel
from infer import generate  # reuse your generate() function

app = FastAPI()

class CodeInput(BaseModel):
    code: str

@app.post("/refactor")
def refactor_code(input: CodeInput):
    return {"refactored": generate(input.code)}
