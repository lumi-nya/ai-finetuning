import torch
import json

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel


class LisaInference:

    def __init__(self, model_name: str, adapter_path: str):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16
        )

        self.model = PeftModel.from_pretrained(base_model, adapter_path)

        self.model.eval()

        print("Model loaded")

    def generate(self, context: list[dict], max_new_tokens: int = 256) -> str:

        # Termina em "\n", igual ao prompt dos exemplos de treino
        # (convert_dataset.py): assim a fronteira entre o prompt e o que o
        # modelo gera cai entre dois tokens, como no treino.
        prompt = "\n".join(
            json.dumps(item, ensure_ascii=False)
            for item in context
        ) + "\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated = outputs[0][
            inputs["input_ids"].shape[1]:
        ]
        return self.tokenizer.decode(generated, skip_special_tokens=True)


def main():

    lisa = LisaInference(
        model_name="Qwen/Qwen3-1.7B-Base",
        adapter_path="./qwen-adapter"
    )

    context = [
        {
            "role": "system",
            "animation_catalog": [
                {
                    "id": "idle_silly_bounce",
                    "description": "The character makes small playful bouncing movements.",
                    "category": "idle",
                    "layer": "body",
                    "tags": ["silly", "playful", "energetic"]
                }
            ]
        },
        {
            "role": "user",
            "text": "What have you been doing today, Lisa?"
        },
    ]

    response = lisa.generate(context)
    print(response)


if __name__ == "__main__":
    main()
    