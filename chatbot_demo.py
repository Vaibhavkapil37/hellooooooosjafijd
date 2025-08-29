from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"  # or "gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add pad token safely
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

def chat():
    print("Chatbot ready! Type 'quit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break
        
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Bot:", response)

chat()
