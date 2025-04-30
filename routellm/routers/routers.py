import abc
import functools
import random
import requests
import csv
import os
import json
import requests
import logging
from typing import Any, Dict, Union, Optional

from litellm import acompletion, completion



# Configure logging to print to the terminal.
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")


import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from routellm.routers.causal_llm.configs import RouterModelConfig
from routellm.routers.causal_llm.llm_utils import (
    load_prompt_format,
    to_openai_api_messages,
)
from routellm.routers.causal_llm.model import CausalLLMClassifier
from routellm.routers.matrix_factorization.model import MODEL_IDS, MFModel
from routellm.routers.similarity_weighted.utils import (
    OPENAI_CLIENT,
    compute_elo_mle_with_tie,
    compute_tiers,
    preprocess_battles,
)


def no_parallel(cls):
    cls.NO_PARALLEL = True

    return cls


class Router(abc.ABC):
    NO_PARALLEL = False

    # Returns a float between 0 and 1 representing the value used to route to models, conventionally the winrate of the strong model.
    # If this value is >= the user defined cutoff, the router will route to the strong model, otherwise, it will route to the weak model.
    @abc.abstractmethod
    def calculate_strong_win_rate(self, prompt):
        pass

    def route(self, prompt, threshold, routed_pair):
        if self.calculate_strong_win_rate(prompt) >= threshold:
            return routed_pair.strong
        else:
            return routed_pair.weak

    def __str__(self):
        return NAME_TO_CLS[self.__class__]


@no_parallel
class CausalLLMRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        score_threshold=4,
        special_tokens=["[[1]]", "[[2]]", "[[3]]", "[[4]]", "[[5]]"],
        num_outputs=5,
        model_type="causal",
        model_id="meta-llama/Meta-Llama-3-8B",
        flash_attention_2=False,
    ):
        model_config = RouterModelConfig(
            model_id=model_id,
            model_type=model_type,
            flash_attention_2=flash_attention_2,
            special_tokens=special_tokens,
            num_outputs=num_outputs,
        )
        prompt_format = load_prompt_format(model_config.model_id)
        self.router_model = CausalLLMClassifier(
            config=model_config,
            ckpt_local_path=checkpoint_path,
            score_threshold=score_threshold,
            prompt_format=prompt_format,
            prompt_field="messages",
            additional_fields=[],
            use_last_turn=True,
        )
        system_message = hf_hub_download(
            repo_id=checkpoint_path, filename="system_ft_v5.txt"
        )
        classifier_message = hf_hub_download(
            repo_id=checkpoint_path, filename="classifier_ft_v5.txt"
        )
        with open(system_message, "r") as pr:
            system_message = pr.read()
        with open(classifier_message, "r") as pr:
            classifier_message = pr.read()
        self.to_openai_messages = functools.partial(
            to_openai_api_messages, system_message, classifier_message
        )

    def calculate_strong_win_rate(self, prompt):
        input = {}
        input["messages"] = self.to_openai_messages([prompt])
        output = self.router_model(input)
        if output is None:
            # Route to strong model if output is invalid
            return 1
        else:
            return 1 - output["binary_prob"]


@no_parallel
class BERTRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        num_labels=3,
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    def calculate_strong_win_rate(self, prompt):
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.numpy()[0]

        exp_scores = np.exp(logits - np.max(logits))
        softmax_scores = exp_scores / np.sum(exp_scores)

        # Compute prob of label 1 and 2 (tie, tier 2 wins)
        binary_prob = np.sum(softmax_scores[-2:])
        return 1 - binary_prob


class SWRankingRouter(Router):
    def __init__(
        self,
        arena_battle_datasets,
        arena_embedding_datasets,
        # This is the model pair for Elo calculations at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        num_tiers=10,
    ):
        self.strong_model = strong_model
        self.weak_model = weak_model

        self.arena_df = concatenate_datasets(
            [load_dataset(dataset, split="train") for dataset in arena_battle_datasets]
        ).to_pandas()
        self.arena_df = preprocess_battles(self.arena_df)

        embeddings = [
            np.array(load_dataset(dataset, split="train").to_dict()["embeddings"])
            for dataset in arena_embedding_datasets
        ]
        self.arena_conv_embedding = np.concatenate(embeddings)
        self.embedding_model = "text-embedding-3-small"

        assert len(self.arena_df) == len(
            self.arena_conv_embedding
        ), "Number of battle embeddings is mismatched to data"

        model_ratings = compute_elo_mle_with_tie(self.arena_df)
        self.model2tier = compute_tiers(model_ratings, num_tiers=num_tiers)

        self.arena_df["model_a"] = self.arena_df["model_a"].apply(
            lambda x: self.model2tier[x]
        )
        self.arena_df["model_b"] = self.arena_df["model_b"].apply(
            lambda x: self.model2tier[x]
        )

    def get_weightings(self, similarities):
        max_sim = np.max(similarities)
        return 10 * 10 ** (similarities / max_sim)

    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        prompt_emb = (
            (
                OPENAI_CLIENT.embeddings.create(
                    input=[prompt], model=self.embedding_model
                )
            )
            .data[0]
            .embedding
        )
        similarities = np.dot(self.arena_conv_embedding, prompt_emb) / (
            np.linalg.norm(self.arena_conv_embedding, axis=1)
            * np.linalg.norm(prompt_emb)
        )

        weightings = self.get_weightings(similarities)
        res = compute_elo_mle_with_tie(self.arena_df, sample_weight=weightings)

        weak_score, strong_score = (
            res[self.model2tier[self.weak_model]],
            res[self.model2tier[self.strong_model]],
        )
        weak_winrate = 1 / (1 + 10 ** ((strong_score - weak_score) / 400))
        strong_winrate = 1 - weak_winrate

        # If the expected strong winrate is greater than the threshold, use strong
        return strong_winrate


@no_parallel
class MatrixFactorizationRouter(Router):
    def __init__(
        self,
        checkpoint_path,
        # This is the model pair for scoring at inference time,
        # and can be different from the model pair used for routing.
        strong_model="gpt-4-1106-preview",
        weak_model="mixtral-8x7b-instruct-v0.1",
        hidden_size=128,
        num_models=64,
        text_dim=1536,
        num_classes=1,
        use_proj=True,
    ):
        device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        self.model = MFModel.from_pretrained(
            checkpoint_path,
            dim=hidden_size,
            num_models=num_models,
            text_dim=text_dim,
            num_classes=num_classes,
            use_proj=use_proj,
        )
        self.model = self.model.eval().to(device)
        self.strong_model_id = MODEL_IDS[strong_model]
        self.weak_model_id = MODEL_IDS[weak_model]

    def calculate_strong_win_rate(self, prompt):
        winrate = self.model.pred_win_rate(
            self.strong_model_id, self.weak_model_id, prompt
        )
        return winrate


# Parallelism makes the randomness non deterministic
@no_parallel
class RandomRouter(Router):
    def calculate_strong_win_rate(
        self,
        prompt,
    ):
        del prompt
        return random.uniform(0, 1)

class OllamaRouter(Router):
    def __init__(self, model_name="llama3", **kwargs):
        """
        Initializes the OllamaRouter.
        :param model_name: The Ollama model to use for routing decisions (e.g., "llama3")
        """
        self.model_name = model_name
        # Set the Ollama API URL – adjust if your endpoint is different
        self.api_url = "http://localhost:11434/api/generate"

    def calculate_strong_win_rate(self, prompt: str) -> float:
        """
        Sends the prompt to the Ollama model and expects an answer indicating whether to use the strong model.
        Returns 1.0 if the answer contains 'strong'; otherwise returns 0.0.
        """
        system_prompt = (
            "You are a routing assistant. Respond ONLY with 'strong' or 'weak' "
            "based on whether the prompt requires a powerful model."
        )

        # Compose the full prompt
        full_prompt = f"{system_prompt}\n\nUser prompt:\n{prompt}"

        # Prepare the payload for the Ollama API request
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
        }

        try:
            response = requests.post(self.api_url, json=payload)
            # Assume the response JSON has a key 'response'
            answer = response.json().get("response", "").strip().lower()
        except Exception as e:
            print(f"[OllamaRouter] Error: {e}")
            answer = "strong"  # fallback to "strong" in case of error

        # Return 1.0 if the model indicates "strong"; otherwise 0.0 to route to the weak model.
        return 1.0 if "strong" in answer else 0.0

class ComplexityAzureRouter(Router):
    def __init__(self, **kwargs):
        """
        Initializes the ComplexityAzureRouter with hard-coded Azure OpenAI configuration.
        """
        # Hard-coded values based on your Azure deployment info.
        self.deployment_id  = os.getenv(DEPLOYMENT_ID_ENV)
        self.resource_name  = os.getenv(RESOURCE_NAME_ENV)
        self.api_version    = os.getenv(API_VERSION_ENV)
        self.azure_api_key  = os.getenv(AZURE_API_KEY_ENV)
        
        # Construct the Azure OpenAI endpoint URL using the chat completions endpoint.
        self.api_url = (
            f"https://{self.resource_name}.openai.azure.com/openai/deployments/"
            f"{self.deployment_id}/chat/completions?api-version={self.api_version}"
        )
        logging.info("ComplexityAzureRouter initialized with endpoint: %s", self.api_url)
        
        # CSV file to log prompt complexity values.
        self.csv_file = "ComplexityAzureRouter.csv"
        if not os.path.isfile(self.csv_file):
            self._initialize_csv()

    def _initialize_csv(self):
        keys = ["wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni", "mu", "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr", "cer", "hs", "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd", "nc", "ifa", "input_tokens", "output_tokens", "avg_score"]

        fieldnames = ["prompt"] + keys + ["avg_score"]
        try:
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            #logging.debug("CSV file initialized with headers: %s", fieldnames)
        except Exception as e:
            logging.error("[ComplexityAzureRouter] Error initializing CSV: %s", e)

    def calculate_strong_win_rate(self, prompt: str) -> float:
        """
        Sends the prompt to the Azure OpenAI model and expects a JSON response with 32 complexity parameters.
        Computes the average of these 32 scores and logs the prompt, individual scores, and average score to a CSV file.
        Returns the average score.
        """
        input_tokens = 0
        output_tokens = 0
        system_prompt = ("You are an expert prompt complexity analyzer (each 1 to 9). Evaluate the  prompt and return a JSON object. Use  abbreviations in output:: wc: word count complexity; ssc: sentence structure complexity; vd: vocabulary diversity; sc: syntactic complexity; sd: semantic depth; ca: conceptual abstraction; lf: logical flow; ac: argumentation complexity; tc: technicality; ni: numerical information; mu: metaphor usage; cr: cultural references; ie: idiomatic expressions; ap: ambiguous phrases; cd: context dependency; it: intertextuality; rq: rhetorical questions; ps: perspective shift; tr: temporal references; spr: spatial references; cer: cause-effect relationship; hs: hypothetical scenarios; crr: counterfactual reasoning; ec: ethical complexity; ei: emotional intensity; hu: humor usage; sl: sarcasm level; cdm: creativity demand; inf: inferential demand; dd: detail density; nc: narrative complexity; ifa: integration of facts.")

        # Build the chat messages required by the Azure chat completions endpoint.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Prompt:\n{prompt}"}
        ]

        # Prepare the payload for Azure OpenAI.
        payload = {
            "messages": messages,
            "max_tokens": 200,  # Adjust as needed based on expected response size.
            "temperature": 0.2,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "response_format": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "complexity_response",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "wc": {"type": "number"},
                                                "ssc": {"type": "number"},
                                                "vd": {"type": "number"},
                                                "sc": {"type": "number"},
                                                "sd": {"type": "number"},
                                                "ca": {"type": "number"},
                                                "lf": {"type": "number"},
                                                "ac": {"type": "number"},
                                                "tc": {"type": "number"},
                                                "ni": {"type": "number"},
                                                "mu": {"type": "number"},
                                                "cr": {"type": "number"},
                                                "ie": {"type": "number"},
                                                "ap": {"type": "number"},
                                                "cd": {"type": "number"},
                                                "it": {"type": "number"},
                                                "rq": {"type": "number"},
                                                "ps": {"type": "number"},
                                                "tr": {"type": "number"},
                                                "spr": {"type": "number"},
                                                "cer": {"type": "number"},
                                                "hs": {"type": "number"},
                                                "crr": {"type": "number"},
                                                "ec": {"type": "number"},
                                                "ei": {"type": "number"},
                                                "hu": {"type": "number"},
                                                "sl": {"type": "number"},
                                                "cdm": {"type": "number"},
                                                "inf": {"type": "number"},
                                                "dd": {"type": "number"},
                                                "nc": {"type": "number"},
                                                "ifa": {"type": "number"}
                                            },
                                            "required": [
                                                "wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni", "mu",
                                                "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr", "cer", "hs",
                                                "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd", "nc", "ifa"
                                            ],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                }
        }

        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key
        }

        # Declare the keys so they are available even if an error occurs.
        keys = ["wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni", "mu", "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr", "cer", "hs", "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd", "nc", "ifa"]

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()

            # Extract the chat message content from the Azure response.
            response_json = response.json()
            generated_text = response_json["choices"][0]["message"]["content"].strip()
            #logging.debug("Received generated text: %s", generated_text)
            # Extract token usage if available.
            usage = response_json.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Parse the generated text as JSON.
            result_json = json.loads(generated_text)
            scores = [float(result_json.get(key, 0)) for key in keys]
            avg_score = sum(scores) / len(scores)
            logging.info("Calculated average score: %f", avg_score)
        except Exception as e:
            logging.error("[ComplexityAzureRouter] Error evaluating prompt complexity: %s", e)
            # Initialize scores to zero for all keys if an error occurs.
            scores = [0 for _ in keys]
            avg_score = 1.0  # Fallback to a strong complexity score.

        # Log the prompt and its complexity scores to the CSV file.
        self._save_to_csv(prompt, keys, input_tokens, output_tokens, scores, avg_score)
        return avg_score

    def _save_to_csv(self, prompt: str, keys: list, input_tokens: int, output_tokens: int, scores: list, avg_score: float):
        """
        Appends a row to the CSV file with the prompt, each of the 32 scores, and the average score.
        """
        fieldnames = ["prompt"] + keys + ["input_tokens", "output_tokens", "avg_score"]
        row = {"prompt": prompt, "avg_score": avg_score}
        for key, score in zip(keys, scores):
            row[key] = score
        row["input_tokens"] = input_tokens
        row["output_tokens"] = output_tokens 
        row["avg_score"] = avg_score
        try:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            #logging.debug("Logged complexity data to CSV for prompt: %s", prompt)
        except Exception as e:
            logging.error("[ComplexityAzureRouter] Error writing CSV: %s", e)

    def route(self, prompt: str, threshold: float, model_pair) -> str:
        """
        Routes the prompt based on the calculated complexity.
        It uses calculate_strong_win_rate to obtain an average score.
        If the score is greater than or equal to the threshold, the strong model is chosen;
        otherwise, the weak model is selected.
        
        :param prompt: The prompt text to evaluate.
        :param threshold: The threshold to decide between models.
        :param model_pair: A ModelPair object with attributes 'strong' and 'weak'.
        :return: The chosen model as a string.
        """
        avg_score = self.calculate_strong_win_rate(prompt)
        #logging.debug("Routing decision: average score %f, threshold %f", avg_score, threshold)
        if avg_score/10 >= threshold:
            logging.info("Routing to strong model: %s", model_pair.strong)
            return model_pair.strong
        else:
            logging.info("Routing to weak model: %s", model_pair.weak)
            return model_pair.weak  

class ComplexitySelfAnswerRouter(Router):
    def __init__(self, **kwargs):
        self.deployment_id  = os.getenv(DEPLOYMENT_ID_ENV)
        self.resource_name  = os.getenv(RESOURCE_NAME_ENV)
        self.api_version    = os.getenv(API_VERSION_ENV)
        self.azure_api_key  = os.getenv(AZURE_API_KEY_ENV)
        self.api_url = (
            f"https://{self.resource_name}.openai.azure.com/openai/deployments/"
            f"{self.deployment_id}/chat/completions?api-version={self.api_version}"
        )
        self.csv_file = "ComplexitySelfAnswerRouter.csv"
        if not os.path.isfile(self.csv_file):
            self._initialize_csv()
        logging.info("ComplexitySelfAnswerRouter initialized with endpoint: %s", self.api_url)

    def _initialize_csv(self):
        keys = ["wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni", "mu", 
                "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr", "cer", "hs", 
                "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd", "nc", "ifa"]
        fieldnames = ["prompt"] + keys + ["input_tokens", "output_tokens", "avg_score"]
        try:
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            #logging.debug("CSV file initialized with headers: %s", fieldnames)
        except Exception as e:
            logging.error("[ComplexitySelfAnswerRouter] Error initializing CSV: %s", e)
    
    def _save_to_csv(self, prompt: str, keys: list, input_tokens: int, output_tokens: int, scores: list, avg_score: float):
        fieldnames = ["prompt"] + keys + ["input_tokens", "output_tokens", "avg_score"]
        row = {"prompt": prompt, "avg_score": avg_score}
        for key, score in zip(keys, scores):
            row[key] = score
        row["input_tokens"] = input_tokens
        row["output_tokens"] = output_tokens 
        row["avg_score"] = avg_score
        try:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            #logging.debug("Logged complexity data to CSV for prompt: %s", prompt)
        except Exception as e:
            logging.error("[ComplexitySelfAnswerRouter] Error writing CSV: %s", e)

    def calculate_complexity_and_answer(self, prompt: str) -> Dict[str, Any]:
        """
        Sends a single request to Azure OpenAI instructing it to evaluate the prompt's complexity and,
        if the prompt is simple, generate an answer.

        The returned JSON is expected to follow a schema that includes 32 complexity metrics plus:
          - answer: string, which will be non-empty if the prompt is answered directly.
          - routeforward: boolean, false if answered, true otherwise.
        """
        logging.info("calculate_complexity_and_answer called for prompt: %s", prompt)
                
        # Build the system prompt.
        system_prompt = "You are a prompt complexity analyzer (each 1 to 9) and chatgpt like answer generator. Evaluate the prompt and return a JSON object. Use abbreviations in output: wc: word count complexity; ssc: sentence structure complexity; vd: vocabulary diversity; sc: syntactic complexity; sd: semantic depth; ca: conceptual abstraction; lf: logical flow; ac: argumentation complexity; tc: technicality; ni: numerical information; mu: metaphor usage; cr: cultural references; ie: idiomatic expressions; ap: ambiguous phrases; cd: context dependency; it: intertextuality; rq: rhetorical questions; ps: perspective shift; tr: temporal references; spr: spatial references; cer: cause-effect relationship; hs: hypothetical scenarios; crr: counterfactual reasoning; ec: ethical complexity; ei: emotional intensity; hu: humor usage; sl: sarcasm level; cdm: creativity demand; inf: inferential demand; dd: detail density; nc: narrative complexity; ifa: integration of facts. As a 2nd step, include the following keys: - answer: ONLY IF (Important) the prompt is simple  include a generated answer string to the user prompt, otherwise set answer to empty; - routeforward: a boolean that is false if an answer is generated and true if the prompt is too complex."
        #logging.debug("System prompt: %s", system_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Prompt:\n{prompt}"}
        ]
        #logging.debug("User message prepared with prompt: %s", prompt)

        payload = {
            "messages": messages,
            "max_tokens": 3000,  # Adjust as needed based on expected response size.
            "temperature": 0.2,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "response_format": {
                                    "type": "json_schema",
                                    "json_schema": {
                                        "name": "complexity_response",
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "wc": {"type": "number"},
                                                "ssc": {"type": "number"},
                                                "vd": {"type": "number"},
                                                "sc": {"type": "number"},
                                                "sd": {"type": "number"},
                                                "ca": {"type": "number"},
                                                "lf": {"type": "number"},
                                                "ac": {"type": "number"},
                                                "tc": {"type": "number"},
                                                "ni": {"type": "number"},
                                                "mu": {"type": "number"},
                                                "cr": {"type": "number"},
                                                "ie": {"type": "number"},
                                                "ap": {"type": "number"},
                                                "cd": {"type": "number"},
                                                "it": {"type": "number"},
                                                "rq": {"type": "number"},
                                                "ps": {"type": "number"},
                                                "tr": {"type": "number"},
                                                "spr": {"type": "number"},
                                                "cer": {"type": "number"},
                                                "hs": {"type": "number"},
                                                "crr": {"type": "number"},
                                                "ec": {"type": "number"},
                                                "ei": {"type": "number"},
                                                "hu": {"type": "number"},
                                                "sl": {"type": "number"},
                                                "cdm": {"type": "number"},
                                                "inf": {"type": "number"},
                                                "dd": {"type": "number"},
                                                "nc": {"type": "number"},
                                                "ifa": {"type": "number"},
                                                "answer": {"type": "string"},
                                                "routeforward": {"type": "boolean"}
                                            },
                                            "required": [
                                                "wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni", "mu",
                                                "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr", "cer", "hs",
                                                "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd", "nc", "ifa", "answer", "routeforward"
                                            ],
                                            "additionalProperties": False
                                        },
                                        "strict": True
                                    }
                                }
        }
        #logging.debug("Payload prepared: %s", payload)

        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key
        }

        keys = [
            "wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni", "mu",
            "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr", "cer", "hs",
            "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd", "nc", "ifa"
        ]

        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()

            # Extract the chat message content from the Azure response.
            response_json = response.json()
            generated_text = response_json["choices"][0]["message"]["content"].strip()
            #logging.debug("Received generated text: %s", generated_text)
            # Extract token usage if available.
            usage = response_json.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            result_json = json.loads(generated_text)
            scores = [float(result_json.get(key, 0)) for key in keys]
            avg_score = sum(scores) / len(scores)
            logging.info("Calculated average score: %f", avg_score)
        except Exception as e:
            logging.error("[ComplexitySelfAnswerRouter] Error evaluating prompt complexity and answer: %s", e)
            result_json = {key: 0 for key in keys}
            result_json["answer"] = ""
            result_json["routeforward"] = True
            avg_score = 1.0
            input_tokens = 0
            output_tokens = 0
            scores = [0 for _ in keys]
        #logging.debug("Saving CSV log for prompt: %s", prompt)
        self._save_to_csv(prompt, keys, input_tokens, output_tokens, scores, avg_score)
        result_json["avg_score"] = avg_score
        result_json["input_tokens"] = input_tokens
        result_json["output_tokens"] = output_tokens
        #logging.debug("Final result_json: %s", result_json)
        return result_json

    def calculate_strong_win_rate(self, prompt: str) -> float:
        """
        Implements the abstract method.
        Uses calculate_complexity_and_answer to get the average score,
        then returns the normalized value (assuming 10 as maximum complexity).
        """
        logging.info("calculate_strong_win_rate called for prompt: %s", prompt)
        result = self.calculate_complexity_and_answer(prompt)
        avg_score = result.get("avg_score", 10.0)
        win_rate = avg_score / 10.0
        logging.info("Strong win rate computed as: %f", win_rate)
        return win_rate

    def route(self, prompt: str, threshold: float, model_pair) -> Union[str, Dict[str, Any]]:
        """
        Returns a JSON object from a single API call that contains both the complexity metrics and,
        if applicable, a self-generated answer. If 'answer' is non-empty, returns the full JSON.
        Otherwise, compares (avg_score/10) with the threshold and returns either model_pair.strong or model_pair.weak.
        """
        logging.info("Routing called for prompt: %s", prompt)
        result = self.calculate_complexity_and_answer(prompt)
        avg_score = result.get("avg_score", 1.0)
        #logging.debug("Routing decision: avg_score=%f", avg_score)
        if "answer" in result and result["answer"].strip():
            logging.info("Self-generated answer available.")
            return result
        else:
            #logging.debug("No self-generated answer. Evaluating threshold: %f", threshold)
            if (avg_score / 10.0) >= threshold:
                logging.info("Routing to strong model: %s", model_pair.strong)
                logging.info("Calculated average score: %f", avg_score)
                return model_pair.strong
            else:
                logging.info("Routing to weak model: %s", model_pair.weak)
                logging.info("Calculated average score: %f", avg_score)
                return model_pair.weak
    
    #Extra
    def calculate_strong_win_rate(self, prompt: str) -> float:
        """
        Implements the abstract method. Uses calculate_complexity_and_answer to get the average score,
        then returns the normalized value (assuming 10 as maximum).
        """
        result = self.calculate_complexity_and_answer(prompt)
        avg_score = result.get("avg_score", 10.0)
        return avg_score / 10.0

class ComplexityEvaluationRouter(Router):
    def __init__(self, **kwargs):
        self.deployment_id = "gpt-4o20240806"
        self.resource_name = "ti-"
        self.api_version = "2024-12-01-preview"
        self.azure_api_key = ""
        self.api_url = (
            f"https://{self.resource_name}.openai.azure.com/openai/deployments/"
            f"{self.deployment_id}/chat/completions?api-version={self.api_version}"
        )
        self.csv_file = "ComplexityEvaluationRouter.csv"
        if not os.path.isfile(self.csv_file):
            self._initialize_csv()
        logging.info("ComplexityEvaluationRouter initialized with endpoint: %s", self.api_url)

    def _initialize_csv(self):
        # 32 complexity keys (same ordering as before)
        complexity_keys = [
            "wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni",
            "mu", "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr",
            "cer", "hs", "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd",
            "nc", "ifa"
        ]
        # 8 evaluation keys (using concise acronyms)
        eval_keys = ["eacc", "erel", "econ", "ecla", "ecoh", "edep", "econc", "estil"]
        fieldnames = ["prompt"] + complexity_keys + ["input_tokens", "output_tokens", "avg_score"] + eval_keys
        try:
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            #logging.debug("CSV file initialized with headers: %s", fieldnames)
        except Exception as e:
            logging.error("[ComplexityEvaluationRouter] Error initializing CSV: %s", e)

    def _save_to_csv(
        self,
        prompt: str,
        complexity_keys: list,
        input_tokens: int,
        output_tokens: int,
        scores: list,
        avg_score: float,
        eval_scores: Dict[str, int]
    ):
        eval_keys = ["eacc", "erel", "econ", "ecla", "ecoh", "edep", "econc", "estil"]
        fieldnames = ["prompt"] + complexity_keys + ["input_tokens", "output_tokens", "avg_score"] + eval_keys
        row = {"prompt": prompt, "avg_score": avg_score}
        for key, score in zip(complexity_keys, scores):
            row[key] = score
        row["input_tokens"] = input_tokens
        row["output_tokens"] = output_tokens
        for key in eval_keys:
            row[key] = eval_scores.get(key, 0)
        try:
            with open(self.csv_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
            #logging.debug("Logged complexity & eval data to CSV for prompt: %s", prompt)
        except Exception as e:
            logging.error("[ComplexityEvaluationRouter] Error writing CSV: %s", e)

    def calculate_complexity_and_answer(self, prompt: str) -> Dict[str, Any]:
        """
        Makes one LLM call instructing the LLM to return:
          1. 32 complexity metrics,
          2. A generated answer (if the prompt is simple),
          3. 8 evaluation (answer) metrics (eacc, erel, econ, ecla, ecoh, edep, econc, estil),
          4. And the keys 'answer' and 'routeforward'.

        DOES NOT write to CSV if routeforward is True.
        Instead, stashes all CSV data in result_json["_csv_data"] for later saving.
        """
        logging.info("calculate_complexity_and_answer called for prompt: %s", prompt)
        system_prompt = (
            "You are a prompt complexity analyzer & answer generator. "
            "For the given prompt, return JSON with keys (values 1-9 unless noted) as follows:\n\n"
            "Complexity (32 keys):\n"
            "  wc: word count, ssc: sent. structure, vd: vocab. diversity, sc: syntactic comp., "
            "sd: semantic depth, ca: concpt. abstraction, lf: logical flow, ac: argumentation, "
            "tc: technicality, ni: numerical info, mu: metaphor usage, cr: cultural refs, ie: idioms, "
            "ap: ambiguous phr., cd: context dep., it: intertext., rq: rhetorical ques., ps: perspect. shift, "
            "tr: temporal refs, spr: spatial refs, cer: cause-effect, hs: hypotheticals, crr: counterfact., "
            "ec: ethics, ei: emotion, hu: humor, sl: sarcasm, cdm: creatv demand, inf: inferential, "
            "dd: detail density, nc: narrative comp., ifa: fact integration;\n\n"
            "Also include:\n"
            "  answer: generated answer string (non-empty if simple), and\n"
            "Eval following params based on answer(1 to 9); if no answer, then mark 0:\n"
            "  eacc: accuracy, erel: relevance, econ: completeness, ecla: clarity, "
            "  ecoh: coherence, edep: depth, econc: conciseness, estil: style;\n\n"
            "routeforward: boolean (false if answer generated, true if not).\n"
            "Return output as JSON."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Prompt:\n{prompt}"}
        ]
        payload = {
            "messages": messages,
            "max_tokens": 3000,
            "temperature": 0.2,
            "top_p": 1.0,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "complexity_eval_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            # 32 complexity keys:
                            "wc": {"type": "number"},
                            "ssc": {"type": "number"},
                            "vd": {"type": "number"},
                            "sc": {"type": "number"},
                            "sd": {"type": "number"},
                            "ca": {"type": "number"},
                            "lf": {"type": "number"},
                            "ac": {"type": "number"},
                            "tc": {"type": "number"},
                            "ni": {"type": "number"},
                            "mu": {"type": "number"},
                            "cr": {"type": "number"},
                            "ie": {"type": "number"},
                            "ap": {"type": "number"},
                            "cd": {"type": "number"},
                            "it": {"type": "number"},
                            "rq": {"type": "number"},
                            "ps": {"type": "number"},
                            "tr": {"type": "number"},
                            "spr": {"type": "number"},
                            "cer": {"type": "number"},
                            "hs": {"type": "number"},
                            "crr": {"type": "number"},
                            "ec": {"type": "number"},
                            "ei": {"type": "number"},
                            "hu": {"type": "number"},
                            "sl": {"type": "number"},
                            "cdm": {"type": "number"},
                            "inf": {"type": "number"},
                            "dd": {"type": "number"},
                            "nc": {"type": "number"},
                            "ifa": {"type": "number"},
                            # Self-answer keys:
                            "answer": {"type": "string"},
                            "routeforward": {"type": "boolean"},
                            # 8 evaluation keys:
                            "eacc": {"type": "number"},
                            "erel": {"type": "number"},
                            "econ": {"type": "number"},
                            "ecla": {"type": "number"},
                            "ecoh": {"type": "number"},
                            "edep": {"type": "number"},
                            "econc": {"type": "number"},
                            "estil": {"type": "number"}
                        },
                        "required": [
                            "wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni", "mu",
                            "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr", "cer", "hs",
                            "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd", "nc", "ifa",
                            "answer", "routeforward",
                            "eacc", "erel", "econ", "ecla", "ecoh", "edep", "econc", "estil"
                        ],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": self.azure_api_key
        }
        complexity_keys = [
            "wc", "ssc", "vd", "sc", "sd", "ca", "lf", "ac", "tc", "ni",
            "mu", "cr", "ie", "ap", "cd", "it", "rq", "ps", "tr", "spr",
            "cer", "hs", "crr", "ec", "ei", "hu", "sl", "cdm", "inf", "dd",
            "nc", "ifa"
        ]
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            generated_text = response_json["choices"][0]["message"]["content"].strip()
            #logging.debug("Received generated text: %s", generated_text)
            usage = response_json.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            result_json = json.loads(generated_text)
            scores = [float(result_json.get(key, 0)) for key in complexity_keys]
            avg_score = sum(scores) / len(scores)
            logging.info("Calculated average score: %f", avg_score)
        except Exception as e:
            logging.error("[ComplexityEvaluationRouter] Error evaluating prompt: %s", e)
            result_json = {key: 0 for key in complexity_keys}
            result_json["answer"] = ""
            result_json["routeforward"] = True
            avg_score = 1.0
            input_tokens = 0
            output_tokens = 0
            scores = [0 for _ in complexity_keys]

        #logging.debug("Saving CSV data internally for prompt: %s", prompt)
        eval_keys = ["eacc", "erel", "econ", "ecla", "ecoh", "edep", "econc", "estil"]
        eval_scores = {key: int(result_json.get(key, 0)) for key in eval_keys}

        # Instead of writing to CSV here, just stash the data in the result
        result_json["_csv_data"] = (
            prompt,
            complexity_keys,
            input_tokens,
            output_tokens,
            scores,
            avg_score,
            eval_scores
        )

        result_json["avg_score"] = avg_score
        result_json["input_tokens"] = input_tokens
        result_json["output_tokens"] = output_tokens
        #logging.debug("Final result_json: %s", result_json)
        return result_json

    def calculate_strong_win_rate(self, prompt: str) -> float:
        logging.info("calculate_strong_win_rate called for prompt: %s", prompt)
        result = self.calculate_complexity_and_answer(prompt)
        avg_score = result.get("avg_score", 10.0)
        win_rate = avg_score / 10.0
        logging.info("Strong win rate computed as: %f", win_rate)
        return win_rate

    def route(self, prompt: str, threshold: float, model_pair: Any) -> Union[str, Dict[str, Any]]:
        logging.info("Routing called for prompt: %s", prompt)
        result = self.calculate_complexity_and_answer(prompt)
        avg_score = result.get("avg_score", 1.0)
        #logging.debug("Routing decision: avg_score=%f", avg_score)

        # If there's an answer => write CSV row now
        if "answer" in result and result["answer"].strip():
            logging.info("Self-generated answer available.")
            csv_data = result.get("_csv_data")
            if csv_data:
                self._save_to_csv(*csv_data)
            return result

        else:
            #logging.debug("No self-generated answer. Evaluating threshold: %f", threshold)
            if (avg_score / 10.0) >= threshold:
                logging.info("Routing to strong model: %s", model_pair.strong)
                return model_pair.strong
            else:
                logging.info("Routing to weak model: %s", model_pair.weak)
                return model_pair.weak

    def completion(self, **kwargs) -> Dict[str, Any]:
        """
        When the prompt is routed to a strong/weak model (i.e. no self‑generated answer),
        modify the outgoing messages to include extra instructions as well as update the
        response_format so that the model returns both an answer and the 8 evaluation metrics.
        """
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            # Append extra instructions.
            extra_instr = {
                "role": "system",
                "content": (
                    "Aditinonally, Evaluate answer based on following properties (1 to 9):\n"
                    "  eacc: accuracy, erel: relevance, econ: completeness, ecla: clarity, "
                    "ecoh: coherence, edep: depth, econc: conciseness, estil: style;\n\n"
                )
            }
            kwargs["messages"].append(extra_instr)
        # Update (or set) the response_format to require the evaluation keys.
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "routed_ans_eval_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "eacc": {"type": "number"},
                        "erel": {"type": "number"},
                        "econ": {"type": "number"},
                        "ecla": {"type": "number"},
                        "ecoh": {"type": "number"},
                        "edep": {"type": "number"},
                        "econc": {"type": "number"},
                        "estil": {"type": "number"}
                    },
                    "required": ["answer", "eacc", "erel", "econ", "ecla", "ecoh", "edep", "econc", "estil"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        params = {"api_base": self.api_base, "api_key": self.azure_api_key}
        if self.api_version:
            params["api_version"] = self.api_version
        params.update(kwargs)
        result = litellm_completion(**params)
        logging.debug("kwargs: %s", kwargs)
        logging.debug("Result dict: %s", result_dict)
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
            logging.debug("CSV file initialized with headers: %s", result_dict)
        else:
            result_dict = result

        csv_data = kwargs.pop("csv_data", None)
        if csv_data and result_dict.get("answer", "").strip():
            self._save_to_csv(*csv_data)    
        
        return result_dict

    async def acompletion(self, **kwargs) -> Dict[str, Any]:
        """
        Async version of completion: update messages and response_format accordingly.
        """
        if "messages" in kwargs and isinstance(kwargs["messages"], list):
            extra_instr = {
                "role": "system",
                "content": (
                     "Aditinonally, Evaluate answer based on following properties (1 to 9):\n"
                    "  eacc: accuracy, erel: relevance, econ: completeness, ecla: clarity, "
                    "ecoh: coherence, edep: depth, econc: conciseness, estil: style;\n\n"
                )
            }
            kwargs["messages"].append(extra_instr)
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "routed_ans_eval_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "eacc": {"type": "number"},
                        "erel": {"type": "number"},
                        "econ": {"type": "number"},
                        "ecla": {"type": "number"},
                        "ecoh": {"type": "number"},
                        "edep": {"type": "number"},
                        "econc": {"type": "number"},
                        "estil": {"type": "number"}
                    },
                    "required": ["answer", "eacc", "erel", "econ", "ecla", "ecoh", "edep", "econc", "estil"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
        params = {"api_base": self.api_base, "api_key": self.azure_api_key}
        if self.api_version:
            params["api_version"] = self.api_version
        params.update(kwargs)
        result = await litellm_acompletion(**params)
        if hasattr(result, "model_dump"):
            result_dict = result.model_dump()
        else:
            result_dict = result
        
        return result_dict

ROUTER_CLS = {
    "random": RandomRouter,
    "mf": MatrixFactorizationRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
    "sw_ranking": SWRankingRouter,
    "ollama_router": OllamaRouter,
    "complexity_AZ_router": ComplexityAzureRouter,
    "complexity_self_answer_router": ComplexitySelfAnswerRouter,
    "complexity_evaluation_router": ComplexityEvaluationRouter
}
NAME_TO_CLS = {v: k for k, v in ROUTER_CLS.items()}
