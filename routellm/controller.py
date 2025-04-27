from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import pandas as pd
import time
import shortuuid  # make sure you have this module installed
from litellm import acompletion, completion
import litellm
from tqdm import tqdm

# Set drop_params if required by your setup
litellm.drop_params = True

# Default configuration for routers augmented using golden label data from GPT-4.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str


class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
        api_version: Optional[str] = None,  # NEW optional parameter
    ):
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.api_version = api_version  # store api_version if provided
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        # Assume ROUTER_CLS is a dict mapping router names to router objects/constructors.
        from routellm.routers.routers import ROUTER_CLS  # adjust the import as needed
        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            self.routers[router] = ROUTER_CLS[router](**config.get(router, {}))

        # Create a chat namespace resembling the OpenAI Python SDK interface.
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        # Expect format: router-[router_name]-[threshold]
        try:
            _, router, threshold_str = model.split("-", 2)
            threshold = float(threshold_str)
        except Exception as e:
            raise RoutingError(f"Invalid model format: {model}. Expected 'router-[router_name]-[threshold]'.") from e
        return router, threshold

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ) -> Union[str, Dict[str, Any]]:
        prompt = messages[-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.model_pair)
        self.model_counts[router][str(routed_model)] += 1
        return routed_model

    # Mainly used for evaluations.
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(router_instance.calculate_strong_win_rate)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(router_instance.calculate_strong_win_rate)
        else:
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)
        return self.routers[router].route(prompt, threshold, self.model_pair)

    def completion(self, *, router: Optional[str] = None, threshold: Optional[float] = None, **kwargs):
        # If model parameter is passed, parse it to extract router and threshold.
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])
        self._validate_router_threshold(router, threshold)

        # Get the routed result.
        routed_result = self._get_routed_model_for_completion(kwargs["messages"], router, threshold)
        # If the routed result includes a direct answer and does not require routing forward, handle it.
        if isinstance(routed_result, dict) and "answer" in routed_result and routed_result["routeforward"] is False:
            return {
                "id": f"chatcmpl-{shortuuid.random()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "self_answer",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": routed_result["answer"]
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": routed_result.get("input_tokens", 0),
                    "completion_tokens": routed_result.get("output_tokens", 0),
                    "total_tokens": routed_result.get("input_tokens", 0) + routed_result.get("output_tokens", 0)
                },
                "complexity": routed_result
            }

        # If routed_result is a dict but no direct answer, extract a model string.
        if isinstance(routed_result, dict):
            model_str = routed_result.get("model_string", None)
            if model_str is None:
                raise RoutingError("Missing model string in routing result.")
            kwargs["model"] = model_str
        else:
            kwargs["model"] = routed_result  # It's already a string.

        # Prepare parameters for the LLM provider.
        params = {"api_base": self.api_base, "api_key": self.api_key}
        if self.api_version:
            params["api_version"] = self.api_version
        params.update(kwargs)
        return completion(**params)

    async def acompletion(self, *, router: Optional[str] = None, threshold: Optional[float] = None, **kwargs):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])
        self._validate_router_threshold(router, threshold)

        routed_result = self._get_routed_model_for_completion(kwargs["messages"], router, threshold)
        # Check for direct answer path.
        if isinstance(routed_result, dict) and "answer" in routed_result and routed_result["routeforward"] is False:
            return {
                "id": f"chatcmpl-{shortuuid.random()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "self_answer",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": routed_result["answer"]
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": routed_result.get("input_tokens", 0),
                    "completion_tokens": routed_result.get("output_tokens", 0),
                    "total_tokens": routed_result.get("input_tokens", 0) + routed_result.get("output_tokens", 0)
                },
                "complexity": routed_result
            }

        # Extract model string if necessary.
        if isinstance(routed_result, dict):
            model_str = routed_result.get("model_string", None)
            if model_str is None:
                raise RoutingError("Missing model string in routing result.")
            kwargs["model"] = model_str
        else:
            kwargs["model"] = routed_result

        params = {"api_base": self.api_base, "api_key": self.api_key}
        if self.api_version:
            params["api_version"] = self.api_version
        params.update(kwargs)
        return await acompletion(**params)
