from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained("./gpt2_tokenizer")
