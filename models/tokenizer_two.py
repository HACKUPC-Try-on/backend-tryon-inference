from transformers import AutoTokenizer


def get_tokenizer_two():
    tokenizer_two = AutoTokenizer.from_pretrained(
        "yisol/IDM-VTON",
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )
    return tokenizer_two
