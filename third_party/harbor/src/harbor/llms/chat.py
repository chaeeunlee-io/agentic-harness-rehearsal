from pathlib import Path
from typing import Any

from harbor.llms.base import BaseLLM, LLMResponse, OutputLengthExceededError
from harbor.models.agent.rollout_detail import RolloutDetail


class Chat:
    def __init__(
        self,
        model: BaseLLM,
        interleaved_thinking: bool = False,
        record_truncated_turns: bool = False,
    ):
        # harness-knobs truncation: when True, a truncated turn is written into
        # the conversation history before the exception propagates (for agents
        # that continue with the truncated text, e.g. the trainable agent).
        # Default False: terminus-2 appends the pair ITSELF in its handler, and
        # recording here too would duplicate the turn.
        self._record_truncated_turns = record_truncated_turns
        self._model = model
        self._messages = []
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._cumulative_cache_tokens = 0
        self._cumulative_cost = 0.0
        self._prompt_token_ids_list: list[list[int]] = []
        self._completion_token_ids_list: list[list[int]] = []
        self._logprobs_list: list[list[float]] = []
        self._extra_list: list[dict[str, Any]] = []
        self._interleaved_thinking = interleaved_thinking
        self._last_response_id: str | None = None

    @property
    def total_input_tokens(self) -> int:
        return self._cumulative_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self._cumulative_output_tokens

    @property
    def total_cache_tokens(self) -> int:
        return self._cumulative_cache_tokens

    @property
    def total_cost(self) -> float:
        return self._cumulative_cost

    @property
    def messages(self) -> list:
        return self._messages

    @property
    def rollout_details(self) -> list[RolloutDetail]:
        """Get detailed rollout information from all chat interactions.

        Returns:
            List containing a single RolloutDetail representing the complete
            linear chat history, or an empty list if no rollout data was collected.

            TODO: consider multiple rollout details for non-linear chat histories, e.g.
            subagents, summarization, etc.
        """
        if (
            not self._prompt_token_ids_list
            and not self._completion_token_ids_list
            and not self._logprobs_list
            and not self._extra_list
        ):
            return []

        rollout_detail: RolloutDetail = {}

        if self._prompt_token_ids_list:
            rollout_detail["prompt_token_ids"] = self._prompt_token_ids_list

        if self._completion_token_ids_list:
            rollout_detail["completion_token_ids"] = self._completion_token_ids_list

        if self._logprobs_list:
            rollout_detail["logprobs"] = self._logprobs_list

        if self._extra_list:
            # Pivot per-turn dicts to per-field lists for consistent indexing
            all_keys = {k for d in self._extra_list for k in d}
            rollout_detail["extra"] = {
                key: [turn.get(key) for turn in self._extra_list]
                for key in sorted(all_keys)
            }

        return [rollout_detail]

    async def chat(
        self,
        prompt: str,
        logging_path: Path | None = None,
        **kwargs,
    ) -> LLMResponse:
        try:
            llm_response: LLMResponse = await self._model.call(
                prompt=prompt,
                message_history=self._messages,
                logging_path=logging_path,
                previous_response_id=self._last_response_id,
                **kwargs,
            )
        except OutputLengthExceededError as exc:
            # harness-knobs truncation: a truncated turn is still a real turn.
            # Record its rollout details and conversation state so callers that
            # choose to continue keep a consistent history (and trainers keep
            # the tokens); then re-raise for callers that choose to fail.
            if exc.llm_response is not None:
                # Token/usage accounting is unconditional: those tokens were
                # genuinely generated and trainers must see them.
                self._accumulate_rollout_details(exc.llm_response)
                usage = exc.llm_response.usage
                if usage is not None:
                    self._cumulative_input_tokens += usage.prompt_tokens
                    self._cumulative_output_tokens += usage.completion_tokens
                    self._cumulative_cache_tokens += usage.cache_tokens
                    self._cumulative_cost += usage.cost_usd
                if self._record_truncated_turns:
                    self._messages.extend(
                        [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": exc.llm_response.content or ""},
                        ]
                    )
            raise

        # Track response chain for Responses API
        if llm_response.response_id is not None:
            self._last_response_id = llm_response.response_id

        # Get token usage and cost from the LLM response
        usage = llm_response.usage
        if usage is not None:
            self._cumulative_input_tokens += usage.prompt_tokens
            self._cumulative_output_tokens += usage.completion_tokens
            self._cumulative_cache_tokens += usage.cache_tokens
            self._cumulative_cost += usage.cost_usd

        # Accumulate rollout details from the response
        self._accumulate_rollout_details(llm_response)

        # Build assistant message with optional reasoning content
        assistant_message = {"role": "assistant", "content": llm_response.content}
        if self._interleaved_thinking and llm_response.reasoning_content:
            assistant_message["reasoning_content"] = llm_response.reasoning_content

        self._messages.extend(
            [
                {"role": "user", "content": prompt},
                assistant_message,
            ]
        )
        return llm_response

    def reset_response_chain(self) -> None:
        """Reset the response chain so the next call sends full message history.

        Call this whenever chat._messages is directly modified (e.g., after
        summarization or unwinding) to ensure the next Responses API call
        doesn't use a stale previous_response_id.
        """
        self._last_response_id = None

    def _accumulate_rollout_details(self, llm_response: LLMResponse) -> None:
        """Accumulate rollout details from an LLM response.

        Args:
            llm_response: The LLM response containing token IDs and logprobs
        """
        # Accumulate prompt token IDs per turn
        if llm_response.prompt_token_ids:
            self._prompt_token_ids_list.append(llm_response.prompt_token_ids)

        # Accumulate completion token IDs per turn
        if llm_response.completion_token_ids:
            # Store completion token IDs for this turn
            self._completion_token_ids_list.append(llm_response.completion_token_ids)

            # Store logprobs for this turn (if available)
            if llm_response.logprobs:
                self._logprobs_list.append(llm_response.logprobs)

        # Accumulate extra provider-specific fields per turn
        if llm_response.extra:
            self._extra_list.append(llm_response.extra)
