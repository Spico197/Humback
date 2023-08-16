# TODO (tzhu): finish predicting with multi-GPUs

import argparse

import torch
from transformers import PreTrainedTokenizer, AutoModelForCausalLM
from fastchat.model import add_model_args, get_conversation_template, load_model


@torch.inference_mode()
def main(args):
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )
    model: AutoModelForCausalLM
    tokenizer: PreTrainedTokenizer

    msg = args.message

    if args.reverse:
        conv = get_conversation_template("vicuna_v1.1_reverse")
    else:
        conv = get_conversation_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], msg)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt])
    inputs = {k: torch.tensor(v).to(args.device) for k, v in inputs.items()}
    output_ids = model.generate(
        **inputs,
        do_sample=True if args.temperature > 1e-5 else False,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )

    print(f"{conv.roles[0]}: {msg}")
    print(f"{conv.roles[1]}: {outputs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--data_filepath", type=str)
    parser.add_argument("--prompt_column_name", type=str, default="instruction")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
