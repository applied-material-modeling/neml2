---
jupyter:
  jupytext:
    cell_metadata_filter: tags,raises-exception
    notebook_metadata_filter: jupytext
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: neml2
    language: python
    name: python3
---

# Use LLM to explain NEML2 input file in natural language


## Overall workflow

1. Generate the NEML2 syntax database
2. Parse an input file to extract objects and parameters
3. Lookup each object's and parameter's description from the syntax database
4. Convert the input file into a structured prompt
5. Instantiate a chat-completion client
6. Post the prompt to the client
7. Receive the response from the chat client

```python
import json
import os
import requests
from pathlib import Path
import subprocess

import neml2
from neml2.reader import describe
from neml2.reader._syntax import SyntaxDB
from neml2.reader._llm import LLMClient
```

## Client

`neml2.reader._llm.LLMClient` defines the protocol for chat-completion client. It can be extended to define custom clients, e.g., OpenAI clients, Anthropic clients, etc.

In this example, we will use Argonne's Argo as the chat client. If you are not on Argonne's internal network, this example will not work. However, you can still use this example as a reference when wrapping your own chat clients.

> For ANL employees, the request body requires a field "user" which should be your ANL domain username

```python
ARGO_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"


class ArgoClient(LLMClient):
    """LLM client backed by the ANL Argo API."""

    def __init__(self, model: str, api_key: str | None):
        self._model = model
        self._api_key = api_key or os.environ["ARGO_API_KEY"]

    def complete(self, system: str, user: str) -> str:
        payload = json.dumps(
            {
                "user": self._api_key,
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stop": [],
                "temperature": 0.1,
                "top_p": 0.9,
                "max_completion_tokens": 64000,
            }
        )
        headers = {"Content-Type": "application/json"}
        response = requests.post(ARGO_URL, data=payload, headers=headers)
        status = response.status_code
        if status != 200:
            raise RuntimeError(f"Argo API request failed with status {status}: {response.text}")
        return response.json()["response"]
```

## Extract the syntax database

```python
# use the neml2-syntax tool to extract the syntax database from the C++ backend
bin = Path(neml2.__path__[0]) / "bin"
result = subprocess.run([str(bin / "neml2-syntax")], capture_output=True, text=True)
syntax = result.stdout
syntax_db = SyntaxDB(syntax=syntax)
```

## Build the prompt

The `describe` method parses the input file and looks up object and parameter descriptions from the syntax database.

```python
prompt = describe("demo_model.i", syntax_db, include_params=True)
```

## Call the LLM chat client

```python
client = ArgoClient(model="gpt54", api_key="thu") # replace with your actual API key
response = client.complete(*prompt)

# render the response as markdown in the notebook
from IPython.display import display, Markdown
display(Markdown(response))
```
