from transformers import AutoTokenizer


def get_tokenizer_one():
    tokenizer_one = AutoTokenizer.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    return tokenizer_one
