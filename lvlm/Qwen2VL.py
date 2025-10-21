import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore")


class Qwen2VL:

    def __init__(self, version):
        self.version = version
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        model_name = f"Qwen/{self.version}"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        model_name,
                        torch_dtype=torch.bfloat16,
                        attn_implementation="flash_attention_2",
                        device_map="auto",
                    )
        self.processor = AutoProcessor.from_pretrained(model_name)

    def _build_messages(self, image, question):
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]

    def _prepare_inputs(self, image, question):
        messages = self._build_messages(image, question)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

    def _dispatch(self, inputs):
        return inputs.to(self.device)

    def generate(self, image, question, temp):
        inputs = self._dispatch(self._prepare_inputs(image, question))
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=32,
            do_sample=True,
            temperature=temp,
            repetition_penalty=1.05,
            top_k=50,
            top_p=0.95,
        )
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answer = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return answer[0]

    @torch.inference_mode()
    def encode_prompt(self, image, question):
        inputs = self._dispatch(self._prepare_inputs(image, question))
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1]
        if hidden_states.ndim != 3:
            raise RuntimeError("Unexpected hidden state shape returned by Qwen2VL model.")
        pooled = hidden_states[:, -1, :].detach().to(torch.float32)
        return pooled.cpu()