# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Callable

import torch
from datasets import load_dataset
from forge.actors.policy import Policy, PolicyConfig, SamplingOverrides, WorkerConfig
from forge.actors.replay_buffer import ReplayBuffer
from forge.controller.actor import ForgeActor
from forge.controller.service import ServiceConfig, shutdown_service, spawn_service
from forge.data.rewards import MathReward, ThinkingReward
from forge.util.metric_logging import get_metric_logger
from monarch.actor import endpoint
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def compute_sequence_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    requires_grad: bool = True,
) -> torch.Tensor:
    context_manager = torch.enable_grad() if requires_grad else torch.no_grad()

    with context_manager:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Apply log softmax to get log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probabilities for the actual tokens (excluding the first token for next-token prediction)
        shifted_input_ids = input_ids[:, 1:]  # Remove first token
        shifted_log_probs = log_probs[:, :-1, :]  # Remove last logit

        # Gather log probabilities for actual tokens
        token_log_probs = torch.gather(
            shifted_log_probs, dim=-1, index=shifted_input_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Sum log probabilities across sequence (masked by attention)
        shifted_attention_mask = attention_mask[:, 1:]
        sequence_log_probs = (token_log_probs * shifted_attention_mask).sum(dim=-1)

        return sequence_log_probs


@dataclass
class Group:
    response: str  # The response text for tokenization
    ref_logprobs: torch.Tensor
    reward: float
    advantage: float = 0.0


class Episode:
    """Episode container for GRPO rollouts."""

    def __init__(self, episode_id: int, prompt: str, target: str, policy_version: int):
        self.episode_id = episode_id
        self.prompt = prompt
        self.target = target
        self.policy_version = policy_version
        self.groups: list[Group] = []

    def add_group(self, group: Group):
        self.groups.append(group)


class Trainer(ForgeActor):
    """GRPO Trainer implementation for policy optimization."""

    def __init__(
        self,
        learning_rate: float = 1e-5,
        beta: float = 0.1,
        model_name: str = "",
        device: torch.device | None = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta = beta  # KL penalty coefficient
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def train_step(self, batch: list[Episode]):
        total_loss = 0.0
        num_groups_processed = 0

        for episode in batch:
            groups = episode.groups

            # Collect all response texts and corresponding data
            response_texts = []
            ref_logprobs_list = []
            advantages_list = []

            for group in groups:
                response_texts.append(group.response)
                ref_logprobs_list.append(group.ref_logprobs)
                advantages_list.append(group.advantage)

            # Tokenize all responses in batch
            tokenized = self.tokenizer(
                response_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,  # Adjust based on your needs
            )

            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)

            # Compute current policy log probabilities using the model
            current_logprobs = compute_sequence_logprobs(
                self.model, input_ids, attention_mask, requires_grad=True
            )

            # Convert ref_logprobs and advantages to tensors
            ref_logprobs_tensor = torch.stack(ref_logprobs_list).to(self.device)
            advantages_tensor = torch.tensor(advantages_list, dtype=torch.float32).to(
                self.device
            )

            # Compute GRPO loss components
            # Ratio between current policy and reference policy
            ratio = torch.exp(current_logprobs - ref_logprobs_tensor)

            # Policy gradient loss weighted by advantages
            pg_loss = -torch.mean(ratio * advantages_tensor)

            # KL penalty to prevent policy from deviating too far from reference
            kl_penalty = self.beta * torch.mean(
                (current_logprobs - ref_logprobs_tensor) ** 2
            )

            # Total GRPO loss
            loss = pg_loss + kl_penalty
            total_loss += loss.item()
            num_groups_processed += len(groups)

            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (optional but recommended for stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

        avg_loss = total_loss / len(batch) if batch else 0.0

        return {"loss": avg_loss, "groups_processed": num_groups_processed}

    @endpoint
    async def update_weights(self, policy_actor):
        """Update policy model weights with trainer's current weights."""
        # Time how long it takes to update weights
        start_time = time.time()

        # Set model to eval mode for weight extraction
        self.model.eval()

        # Extract current model state dict
        model_state_dict = self.model.state_dict()

        # Convert tensors to CPU for transfer (if they're on GPU)
        cpu_state_dict = {}
        for key, tensor in model_state_dict.items():
            cpu_state_dict[key] = tensor.cpu() if tensor.is_cuda else tensor

        # Update the policy actor's model weights
        await policy_actor.update_model_weights.choose(cpu_state_dict)

        # Set model back to training mode
        self.model.train()

        # Log the time taken
        end_time = time.time()
        self.logger.info(f"Updating weights took {end_time - start_time:.2f} seconds")


class RewardActor(ForgeActor):
    """Reward actor that uses a list of scoring functions."""

    def __init__(self, reward_functions: list[Callable]):
        super().__init__()
        self.reward_functions = reward_functions

    @endpoint
    async def evaluate_response(self, prompt: str, response: str, target: str) -> float:
        total_reward = 0.0
        for reward_fn in self.reward_functions:
            reward = reward_fn(prompt, response, target)
            total_reward += reward
        return total_reward


class ComputeAdvantages(ForgeActor):
    """Compute advantages for GRPO using reward signals."""

    def __init__(self, gamma: float = 0.99, lambda_: float = 0.95):
        super().__init__()
        self.gamma = gamma  # Discount factor
        self.lambda_ = lambda_  # GAE lambda parameter

    @endpoint
    async def __call__(self, groups: list[Group]) -> list[float]:
        # Extract rewards from groups
        rewards = [group.reward for group in groups]
        num_groups = len(groups)

        # For simplicity, use reward-to-go as advantages
        # This is a valid advantage estimator: A(s,a) = Q(s,a) - V(s)
        # where Q(s,a) ≈ reward-to-go and V(s) ≈ average reward

        # Compute discounted reward-to-go for each step
        reward_to_go = []
        running_reward = 0.0

        # Calculate discounted returns (reward-to-go)
        for t in reversed(range(num_groups)):
            running_reward = rewards[t] + self.gamma * running_reward
            reward_to_go.insert(0, running_reward)

        # Compute baseline (mean of rewards) and advantages
        baseline = sum(rewards) / len(rewards) if rewards else 0.0
        advantages = [rtg - baseline for rtg in reward_to_go]

        # Normalize advantages to have zero mean and unit variance
        if len(advantages) > 1:
            mean_adv = sum(advantages) / len(advantages)
            var_adv = sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)
            std_adv = (var_adv**0.5) if var_adv > 1e-8 else 1.0
            advantages = [(a - mean_adv) / std_adv for a in advantages]

        return advantages


class RefModel(ForgeActor):
    def __init__(self, model_name, device: torch.device | None = None):
        super().__init__()
        self.model_name = model_name

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        # Set model to eval mode for reference computations
        self.model.eval()

        self.logger.info(f"Model initialized on {self.device}")

    @endpoint
    async def forward(self, token_ids: list[int]) -> torch.Tensor:
        # Use provided token_ids directly
        input_ids = (
            torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        )
        # Create attention mask of all 1s since we have actual tokens (no padding)
        attention_mask = torch.ones_like(input_ids).to(self.device)

        # Compute log probabilities using shared utility function
        sequence_log_probs = compute_sequence_logprobs(
            self.model, input_ids, attention_mask, requires_grad=False
        )

        return (
            sequence_log_probs.squeeze()
        )  # Remove batch dimension for single response


class DatasetActor(ForgeActor):
    """Actor wrapper for HuggingFace dataset to provide async interface."""

    def __init__(
        self, path: str, config_name: str, split: str, streaming: bool, **kwargs
    ):
        super().__init__()

        def gsm8k_to_messages(sample):
            question = sample["question"]
            full_answer: str = sample["answer"]
            answer = full_answer.split("#### ")[1]
            return {"question": question, "answer": answer}

        ds = load_dataset(path, config_name, split=split, streaming=streaming)
        ds = ds.map(gsm8k_to_messages)
        ds = ds.shuffle()
        self._iterator = iter(ds)

    @endpoint
    async def __next__(self) -> dict[str, str] | None:
        try:
            return next(self._iterator)
        except StopIteration:
            return None


async def main():
    """Main GRPO training loop with rollout and training processes."""
    group_size = 1
    model = "Qwen/Qwen3-1.7B"

    # ---- Setup WandB Logger ---- #
    logger = get_metric_logger(
        "wandb",
        freq=1,
        project="grpo-training",
    )

    # ---- Setup services ---- #
    (
        dataloader,
        policy,
        trainer,
        replay_buffer,
        compute_advantages,
        ref_model,
        reward_actor,
    ) = await asyncio.gather(
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            DatasetActor,
            path="openai/gsm8k",
            config_name="main",
            split="train",
            streaming=True,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=1),
            Policy,
            config=PolicyConfig(
                worker_params=WorkerConfig(model=model),
                sampling_params=SamplingOverrides(
                    num_samples=group_size, max_tokens=16
                ),
            ),
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, with_gpus=True, num_replicas=1),
            Trainer,
            learning_rate=1e-5,
            beta=0.1,
            model_name=model,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            ReplayBuffer,
            batch_size=4,
            max_policy_age=1,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            ComputeAdvantages,
            gamma=0.99,
            lambda_=0.95,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1, with_gpus=True),
            RefModel,
            model_name=model,
        ),
        spawn_service(
            ServiceConfig(procs_per_replica=1, num_replicas=1),
            RewardActor,
            reward_functions=[MathReward(), ThinkingReward()],
        ),
    )

    print("All services initialized successfully!")

    # ---- Core RL loops ---- #
    async def continuous_rollouts():
        rollout_count = 0
        while True:
            sample = await dataloader.__next__.choose()
            if sample is None:
                print("Dataloader is empty, exiting continuous rollout")
                return
            prompt, target = sample["question"], sample["answer"]
            version = 0  # await policy.get_current_version.choose()
            episode = Episode(
                episode_id=rollout_count,
                prompt=prompt,
                target=target,
                policy_version=version,
            )
            actions = await policy.generate.choose(prompt)
            for action in actions:
                ref_logprobs = await ref_model.forward.choose(action.token_ids)
                reward = await reward_actor.evaluate_response.choose(
                    prompt=prompt, response=action.text, target=target
                )
                episode.add_group(
                    Group(
                        response=action.text,
                        ref_logprobs=ref_logprobs,
                        reward=reward,
                    )
                )

            advantages = await compute_advantages.__call__.choose(episode.groups)
            for advantage, group in zip(advantages, episode.groups):
                group.advantage = advantage

            await replay_buffer.add.choose(episode)

            rollout_count += 1
            if rollout_count % 10 == 0:
                avg_reward = sum(group.reward for group in episode.groups) / len(
                    episode.groups
                )
                print(
                    f"Generated {rollout_count} rollouts w/ average reward {avg_reward}"
                )
                logger.log("reward/rollout", avg_reward, rollout_count)

    async def continuous_training():
        training_step = 0
        while True:
            batch = await replay_buffer.sample.choose(curr_policy_version=0)
            if batch is None:
                await asyncio.sleep(0.1)
            else:
                training_result = await trainer.train_step.choose(batch)
                training_step += 1
                if training_step % 10 == 0:
                    print(f"Completed {training_step} training steps")
                    if training_result:
                        loss_value = training_result.get("loss", 0.0)
                        print(f"Latest loss: {loss_value}")
                        logger.log("loss/training_step", loss_value, training_step)
                # await trainer.update_weights(policy)

    print("Starting GRPO training loops...")
    rollout_task = asyncio.create_task(continuous_rollouts())
    training_task = asyncio.create_task(continuous_training())

    try:
        await asyncio.gather(rollout_task, training_task)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        rollout_task.cancel()
        training_task.cancel()
    finally:
        print("Shutting down...")
        await asyncio.gather(
            shutdown_service(policy),
            shutdown_service(trainer),
            shutdown_service(replay_buffer),
            shutdown_service(dataloader),
            shutdown_service(compute_advantages),
            shutdown_service(ref_model),
            shutdown_service(reward_actor),
        )


if __name__ == "__main__":
    asyncio.run(main())
