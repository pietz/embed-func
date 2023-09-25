import azure.functions as func
from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
import logging
import json

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)


@app.route(route="embed")
def embed(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    session = ort.InferenceSession(
        "bge-small-en-v1.5.onnx",
        providers=["AzureExecutionProvider", "CPUExecutionProvider"],
    )
    tokenizer = Tokenizer.from_file("tokenizer.json")

    logging.info("Loading model and tokenizer")

    req_body = req.get_json()
    docs = req_body.get("docs")

    embeddings = []
    for doc in docs:
        tok = tokenizer.encode(doc)
        inp = {
            "input_ids": np.array(tok.ids, dtype=np.int64)[None, ...],
            "attention_mask": np.array(tok.attention_mask, dtype=np.int64)[None, ...],
            "token_type_ids": np.array(tok.type_ids, dtype=np.int64)[None, ...],
        }
        embed = session.run(None, inp)[0][0, 0]
        if req_body.get("normalize"):
            embed = embed / np.linalg.norm(embed)
        embeddings.append([round(x, 4) for x in embed.tolist()])

    return json.dumps(embeddings)
