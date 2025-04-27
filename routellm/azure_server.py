"""
Azure OpenAI-Compatible API Server using LiteLLM and Routellm Controller

This server provides OpenAI-compatible RESTful APIs for chat completions (and other endpoints, if needed)
that target Azure OpenAI deployments. It uses routellm’s Controller for routing requests and supports
Azure-specific parameters via command-line options or environment variables.

Usage example (with environment variables set):
    export AZURE_API_KEY="your-api-key"
    export AZURE_API_BASE="https://example-endpoint.openai.azure.com"
    export AZURE_API_VERSION="2023-05-15"

    python -m routellm.azure_server --verbose --port 6060
"""

import argparse
import json
import logging
import os
import time
from collections import defaultdict
from typing import AsyncGenerator, Dict, List, Literal, Optional, Union

import fastapi
import shortuuid
import uvicorn
import yaml
from fastapi.concurrency import asynccontextmanager
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from routellm.controller import Controller, RoutingError
from routellm.routers.routers import ROUTER_CLS

# Disable tokenizer parallelism if needed.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
CONTROLLER = None

# (Optionally, you can initialize an async Azure client here if needed.)
count = defaultdict(lambda: defaultdict(int))

@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    global CONTROLLER
    # Load the YAML config if provided.
    config = yaml.safe_load(open(args.config, "r")) if args.config else None

    # Create the Controller for request routing.
    # Note: The API version is passed for Azure deployments.
    CONTROLLER = Controller(
        routers=args.routers,
        config=config,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
        api_base=args.base_url,
        api_key=args.api_key,
        api_version=args.api_version,  # Pass Azure API version here
        progress_bar=True,
    )
    yield
    CONTROLLER = None

app = fastapi.FastAPI(lifespan=lifespan)

# ------------------- Pydantic Request/Response Models ------------------- #

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length"]] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{shortuuid.random()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo

class ChatCompletionRequest(BaseModel):
    # OpenAI fields: https://platform.openai.com/docs/api-reference/chat/create
    model: str
    messages: Union[
        str,
        List[Dict[str, str]],
        List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ]
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[int, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    tools: Optional[List[Dict[str, Union[str, int, float]]]] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None

# ------------------- Helper Function for Streaming ------------------- #

async def stream_response(response) -> AsyncGenerator:
    import json  # ensure json is imported
    async for chunk in response:
        # If chunk has model_dump_json, use it; otherwise, dump the chunk
        if hasattr(chunk, "model_dump_json"):
            yield f"data: {chunk.model_dump_json()}\n\n"
        else:
            yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"

# ------------------- Endpoint Implementations ------------------- #

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    logging.info(f"Received chat completion request: {request}")
    try:
        # Use model_dump to get a dictionary if available, otherwise assume it's already a dict.
        request_data = (
            request.model_dump(exclude_none=True)
            if hasattr(request, "model_dump")
            else request.dict(exclude_none=True)
        )
        res = await CONTROLLER.acompletion(**request_data)
    except RoutingError as e:
        return JSONResponse(
            ErrorResponse(message=str(e)).model_dump()
            if hasattr(ErrorResponse, "model_dump")
            else ErrorResponse(message=str(e)).dict(),
            status_code=400,
        )

    logging.info(CONTROLLER.model_counts)

    if request.stream:
        return StreamingResponse(
            content=stream_response(res), media_type="text/event-stream"
        )
    else:
        content = res.model_dump() if hasattr(res, "model_dump") else res
        return JSONResponse(content=content)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(content={"status": "online"})

# ------------------- Command-Line Argument Parsing & Uvicorn Launch ------------------- #

parser = argparse.ArgumentParser(
    description="Azure OpenAI-Compatible API Server for LLM routing using LiteLLM"
)
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--workers", type=int, default=0)
parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")
parser.add_argument("--port", type=int, default=6060)
parser.add_argument(
    "--routers",
    nargs="+",
    type=str,
    default=["random"],  # Default router(s)
    choices=list(ROUTER_CLS.keys()),
)
parser.add_argument(
    "--base-url",
    help="The base URL used for all LLM requests (Azure API Base URL)",
    type=str,
    default=None,
)
parser.add_argument(
    "--api-key",
    help="The API key used for all LLM requests (Azure API Key)",
    type=str,
    default=None,
)
parser.add_argument(
    "--api-version",
    help="The API version used for Azure OpenAI requests",
    type=str,
    default=None,
)
parser.add_argument("--strong-model", type=str, default="azure/o1-mini-12092024")
parser.add_argument("--weak-model", type=str, default="azure/o3-mini-31012025")
args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    print("Launching Azure OpenAI server with routers:", args.routers)
    uvicorn.run(
        "routellm.azure_server:app",
        host="0.0.0.0",
        port=args.port,
        workers=args.workers,
    )
