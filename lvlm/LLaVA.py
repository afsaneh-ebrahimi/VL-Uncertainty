import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import warnings
warnings.filterwarnings("ignore")


class LLaVA:

    def __init__(self, version):
        self.version = version
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        model_name = f"llava-hf/{self.version}"
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation='flash_attention_2',
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _prepare_inputs(self, image, question):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image"
                    }
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=image, text=prompt, return_tensors='pt')
        return inputs

    def generate(self, image, question, temp):
        inputs = self._prepare_inputs(image, question).to(self.device, torch.float16)
        output = self.model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=temp,
        )
        final_ans = self.processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT: ')[-1].strip()
        return final_ans

    @torch.inference_mode()
    def encode_prompt(self, image, question):
        inputs = self._prepare_inputs(image, question).to(self.device, torch.float16)
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        if hidden_states.ndim != 3:
            raise RuntimeError("Unexpected hidden state shape returned by LLaVA model.")
        pooled = hidden_states[:, -1, :].detach().to(torch.float32)
        return pooled.cpu()