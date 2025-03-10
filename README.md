# 🎉 Fine-Tuning Your DeepSeek-R1 Towards Medical Experts 🚀

Welcome to the DeepSeek-R1 fine-tuning repository! 🩺 This repository provides an easy-to-use pipeline for fine-tuning the **DeepSeek-R1** model for medical tasks. You can try it out directly on Google Colab and experience the joy of fine-tuning the model. 😄

## 💻 Try It Now on Colab!
Click the link below to open the Google Colab notebook, where you can directly run and experiment with the code. Let's make fine-tuning DeepSeek-R1 for medical tasks fun and easy! 🎉

👉 [Open Colab Notebook](https://colab.research.google.com/drive/1jmCFmgb0DlHbn8SQth-l-r2PMNIZnKLI?usp=sharing) 📑

---

## 🚀 Features
- **Medical Question Answering**: Fine-tuned on medical reasoning tasks, with the ability to answer clinical questions. 🤖
- **Efficient Training**: Uses the unsloth library for memory-efficient training with 4-bit quantization. 💡
- **Fast Inference**: 2x faster inference, allowing for quick and responsive interactions. ⚡
- **Colab Ready**: Easily accessible through Google Colab for hands-on experience. 💻

---

## 🛠 How to Use

### 1. Set Up Dependencies
To get started, you need to install the required libraries. You can do this using the following commands:

```bash
!pip install --no-deps unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
!pip install bitsandbytes unsloth_zoo
!pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
!pip install --no-deps cut_cross_entropy unsloth_zoo
!pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
```

### 2. Load Pre-trained Model
Load the **DeepSeek-R1** model using the unsloth library, which comes pre-trained and ready to fine-tune.

```python
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
    max_seq_length = 2048,
    dtype = None,  # Auto-detect data type
    load_in_4bit = True  # Use 4-bit quantization
)
```

### 3. Fine-tuning with Custom Medical Prompts
Customize the model with your own medical prompts to suit your specific tasks.

```python
train_prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""
```

### 4. Training and Evaluation
Once your dataset is prepared, you can fine-tune the model using the following code:

```python
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,  # Modify if training for more epochs
        learning_rate = 2e-4,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        output_dir = "outputs"
    ),
)
trainer.train()
```

### 5. Inference Example
Once your model is trained, you can use it to generate answers for medical questions.

```python
question = "一个患有急性阑尾炎的病人已经发病5天，腹痛稍有减轻但仍然发热，在体检时发现右下腹有压痛的包块，此时应如何处理？"
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=1200, use_cache=True)
response = tokenizer.batch_decode(outputs)
print(response[0].split("### Response:")[1])
```

## 🤝 Contributing
We welcome contributions! 🎉 Feel free to open issues, submit pull requests, or suggest new ideas. Contributions that enhance the model’s medical reasoning abilities or improve performance are especially encouraged! 💡

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 📄
