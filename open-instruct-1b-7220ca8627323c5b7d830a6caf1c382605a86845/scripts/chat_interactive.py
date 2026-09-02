#!/usr/bin/env python
"""Interactive chat script for testing DPO-tuned models."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with a trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="output/olmo2_1b_dpo_deepspeed/olmo2_1b_dpo__123__1770227059",
        help="Path to the model (local path or HuggingFace repo)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda/cpu)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0 for greedy)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for the conversation",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map=args.device,
    )
    model.eval()
    print(f"Model loaded on {args.device}")

    # Initialize conversation history
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print("\n" + "=" * 60)
    print("Interactive Chat")
    print("=" * 60)
    print(f"System: {args.system_prompt}")
    print("Type 'quit' or 'exit' to end, 'clear' to reset conversation")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
            messages = []
            if args.system_prompt:
                messages.append({"role": "system", "content": args.system_prompt})
            print("[Conversation cleared]\n")
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(args.device)

        with torch.no_grad():
            if args.temperature == 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

        # Decode only the new tokens
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        ).strip()

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})

        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    main()
