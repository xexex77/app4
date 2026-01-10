from __future__ import annotations

import argparse

import torch

from app4.ttt.model.configs import get_config
from app4.ttt.model.llama_ttt import TTTLlamaForCausalLM


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="bringup_1p3b")
    p.add_argument("--prompt-tokens", type=str, default="1,2,3")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eos", type=int, default=-1)
    return p.parse_args()


@torch.no_grad()
def main():
    a = parse_args()
    cfg = get_config(a.config)
    device = torch.device(a.device)

    model = TTTLlamaForCausalLM(cfg).to(device=device)
    model.eval()

    prompt = [int(x.strip()) for x in a.prompt_tokens.split(",") if x.strip()]
    input_ids = torch.tensor([prompt], device=device, dtype=torch.long)

    cache = model.init_cache(
        batch_size=1,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )

    # prefill: dual for efficiency
    logits, cache = model(input_ids, cache=cache, use_dual=True, checkpoint_ttt=False)

    out = prompt[:]
    for _ in range(a.max_new_tokens):
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        out.append(next_id)

        if a.eos >= 0 and next_id == a.eos:
            model.reset_cache(cache)
            break

        # decode: primal sequential (b=1)
        logits, cache = model(
            torch.tensor([[next_id]], device=device, dtype=torch.long),
            cache=cache,
            use_dual=False,
            checkpoint_ttt=False,
        )

    print(",".join(map(str, out)))


if __name__ == "__main__":
    main()

