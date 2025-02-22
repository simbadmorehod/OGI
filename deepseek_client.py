import os
import json
import logging
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekClient:
    def __init__(self, model_path="models/DeepSeek-R1-Distill-Qwen-1.5B-fp16",
                 hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        self.model_path = model_path
        self.hf_model_name = hf_model_name
        self.device = self._get_best_device()
        print(f"Модель загружена на устройство: {self.device}")

        if not self._is_model_downloaded():
            print(f"🔍 Модель не найдена в {model_path}. Скачиваем с Hugging Face...")
            self._download_model()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})

        print(f"✅ Загружаем модель из {model_path} на {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,  # Критически важная опция
            attn_implementation="eager"  # Отключаем оптимизации внимания
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        print(f"🔍 Текущая конфигурация модели: {self.model.config}")

    def _get_best_device(self):
        # Принудительно использовать CPU для тестирования
        return "cpu"  # Заменить на "mps" после проверки
        # if torch.cuda.is_available():
        #     return "cuda"
        # elif torch.backends.mps.is_available():
        #     return "mps"
        # else:
        #     return "cpu"


    def _is_model_downloaded(self) -> bool:
        required_files = ["config.json", "pytorch_model.bin", "tokenizer_config.json"]
        return os.path.exists(self.model_path) and all(
            os.path.exists(os.path.join(self.model_path, f)) for f in required_files
        )

    def _download_model(self):
        print(f"⏳ Скачиваем модель {self.hf_model_name} с Hugging Face...")
        try:
            snapshot_download(repo_id=self.hf_model_name, local_dir=self.model_path)
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                if "quantization_config" in config:
                    del config["quantization_config"]
                with open(config_path, "w") as f:
                    json.dump(config, f, indent=4)
            print(f"✅ Модель успешно скачана в {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка при скачивании модели: {str(e)}")
            raise

    def analyze_query(self, question: str) -> dict:
        """
        Анализ запроса, генерация JSON с ключевыми словами.
        """
        prompt = f"""
        You are a JSON generator.
        Return strictly ONE valid JSON object with fields "keywords" (array of strings) and "time_filter" (string).
        Example:
        {{"keywords":["BTC"],"time_filter":"last_week"}}
        Now output JSON for: {question}
        """

        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return json.loads(response)


if __name__ == "__main__":
    client = DeepSeekClient()
    question = "Какие новости о BTC за последнюю неделю?"
    result = client.analyze_query(question)
    print(result)