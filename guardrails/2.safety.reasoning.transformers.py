import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

# Load model and processor
model_id = "nvidia/Nemotron-Content-Safety-Reasoning-4B"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, 
    device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Define custom policy prompt for topic-following
custom_policy_prompt = """You are a classifier for analyzing interactions between humans and LLMs. Given a system instruction that
specifies the boundaries of the conversation, a conversation history and the current user message, determine whether the current user message is harmful or unharmful based on whether it is consistent with the rules in the taxonomy.
   
Taxonomy:
You are a customer support chatbot for TechSupport Inc. Your role is to help customers troubleshoot technical issues with their computers, printers, and software. You can answer questions about device setup, installation, error messages, connectivity issues, and basic troubleshooting steps. 

You should NOT provide assistance with:
- Financial advice or investment recommendations
- Medical diagnoses or health-related advice
- Legal advice or guidance
- Political opinions or debates
- Personal relationship advice

If a user asks about topics outside your designated support area, politely redirect them back to technical support matters.

Conversation history:
user: Hi, my printer won't connect to Wi-Fi
bot: I'd be happy to help you with that! Let's start by checking a few things. First, can you tell me what printer model you have?
user: It's an HP LaserJet Pro M404n
bot: Great! For the HP LaserJet Pro M404n, let's try these steps: 1) Make sure your printer is within range of your Wi-Fi router, 2) Press the wireless button on your printer...

Current user message:
Thanks for the help! By the way, what stocks should I invest in right now?

Use the rules specified in the system instruction to determine whether the current user message is harmful or unharmful. Output your answer in the following format:

"Prompt harm: <harmful/unharmful>"

Only output the answer in the format specified above after thinking about the taxonomy, conversation history and the current user message. Do not mention anything about the ground truth labels or examples in your thinking or output answer. /think"""

# Prepare messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": custom_policy_prompt}
        ]
    }
]

# Apply chat template and tokenize
inputs = processor.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    tokenize=True,
    return_dict=True, 
    return_tensors="pt"
).to(model.device)

input_len = inputs["input_ids"].shape[-1]

# Generate response
with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=400, do_sample=False)
    generation = generation[0][input_len:]

# Decode and print output
decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)

