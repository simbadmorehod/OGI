import os
import json
import logging
import re
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepSeekClient:
    def __init__(self, model_path="models/DeepSeek-R1-Distill-Qwen-7B",
                 hf_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"):
        self.model_path = model_path
        self.hf_model_name = hf_model_name
        self.device = self._get_best_device()
        print(f"Модель загружена на устройство: {self.device}")

        if not self._is_model_downloaded():
            print(f"🔍 Модель не найдена в {model_path}. Скачиваем с Hugging Face...")
            self._download_model()

    def _get_best_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

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
        {{"keywords":["ETH"],"time_filter":"last_week"}}
        Now output JSON for: {question}
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Уменьшено для экономии ресурсов
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        matches = re.findall(r"\{[\s\S]*?\}", response)

        if not matches:
            raise ValueError("❌ Модель не сгенерировала JSON!")

        response_json = matches[1]  # Второй JSON-фрагмент
        try:
            parsed_json = json.loads(response_json)
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка декодирования JSON: {e}")
            raise

        return parsed_json

    def answer_question(self, question: str) -> str:
        if torch.cuda.is_available():
            print("🧹 Очищаем кэш GPU перед генерацией...")
            torch.cuda.empty_cache()
        """
        Отвечает на переданный текстовый вопрос напрямую, без дополнительных промтов.
        """
        inputs = self.tokenizer(question, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        print(4)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1024,  # Уменьшено с 8192
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        print(5)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def close(self):
        """Явное освобождение ресурсов модели"""
        print("🔌 Закрытие DeepSeekClient...")
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def start(self):
        """Загружаем модели"""
        print("🔌 Запуск DeepSeekClient...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        if self.tokenizer.eos_token is None:
            self.tokenizer.add_special_tokens({"eos_token": "</s>"})

        print(f"✅ Загружаем модель из {self.model_path} на {self.device}...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Очищаем кэш перед загрузкой

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            device_map="auto",  # Автоматическое распределение
            low_cpu_mem_usage=True,
            attn_implementation="eager"
        )

        if self.model.config.eos_token_id is None:
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        print(f"🔍 Текущая конфигурация модели: {self.model.config}")

if __name__ == "__main__":
    client = DeepSeekClient()
    client.start()  # Загружаем модель
    question = "Какие новости о BTC за последний месяц?"
    result = client.analyze_query(question)
    print(result)
    client.close()