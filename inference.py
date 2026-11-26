import json

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from peft import PeftModel

from prompting import DEFAULT_MOOD, render_prompt


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

    def generate(self, context: list[dict], mood: dict | None = None, max_new_tokens: int = 256) -> str:

        # Igual ao prompt dos exemplos de treino (ver prompting.py): o mood atual
        # é a última linha; sem branch e sem tool_call/tool_result de turnos
        # passados; termina em "\n".
        prompt = render_prompt(context, mood=mood)

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


    def respond(self, context: list[dict], mood: dict, tool_handler, max_new_tokens: int = 256, max_steps: int = 4) -> list[dict]:
        """
        Gera a resposta de um turno, executando tools no caminho.

        `context` termina na mensagem 'user' do turno; `mood` é o estado atual
        (entrada do modelo, vai no fim do prompt). `tool_handler(tool_call)`
        recebe a mensagem tool_call (`name`, `arguments`) e devolve o conteúdo do
        tool_result (qualquer valor JSON, por exemplo {"found": false}).

        Retorna as mensagens do turno, na ordem: tool_call e tool_result (se
        houve) e o answer. Quem aplica o `mood_delta` do answer ao mood é o
        programa (prompting.apply_mood_delta, ou session.Session).
        """
        context = list(context)
        produced = []
        decoder = json.JSONDecoder()

        for _ in range(max_steps):
            text = self.generate(context, mood, max_new_tokens)
            messages, pos = [], 0
            while True:
                start = text.find("{", pos)
                if start == -1:
                    break
                try:
                    obj, pos = decoder.raw_decode(text, start)
                except ValueError:
                    break
                if not isinstance(obj, dict) or obj.get("role") in ("user", "system", "tool_result"):
                    break  # o modelo começou a inventar o que não é dele
                messages.append(obj)

            calls = [m for m in messages if m.get("role") == "tool_call"]
            answers = [m for m in messages if m.get("role") == "answer"]
            if answers or not calls:
                return produced + answers[:1]

            for call in calls:
                result = {"role": "tool_result", "name": call.get("name"), "content": tool_handler(call)}
                context += [call, result]
                produced += [call, result]

        raise RuntimeError(f"O modelo pediu tools por mais de {max_steps} passos seguidos.")


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

    response = lisa.generate(context, mood=DEFAULT_MOOD)
    print(response)


if __name__ == "__main__":
    main()
    