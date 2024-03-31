import os
from abc import ABC, abstractmethod
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime
import time
import pandas as pd
from utils.config.GenerationConfig import quantization_config
    
    
class GenerativeModel(ABC):
    def __init__(self,
                 model_name,
                 system_prompt=None):
        
        self.model_name = model_name
        self.system_prompt = system_prompt

        
    @abstractmethod
    def inference(self,
                  text):
        pass

    @abstractmethod
    def load(self):
        pass
    
    @abstractmethod
    def config_prompt(self):
        pass

    
class GigaApi(GenerativeModel):
    def __init__(self,
                 model_name,
                 system_prompt='',
                 credentials = ''):
        
        super().__init__(model_name, system_prompt)
        self.credentials = credentials
        self.messages = []
        self.chat = None
        self.load()

        
    def load(self):
        """Load the model and tokenizer."""
        self.chat = GigaChat(model="GigaChat-Pro", credentials=self.credentials, verify_ssl_certs=False)


    def config_prompt(self, system_prompt):
        """Configure or update the system prompt."""
        self.system_prompt = system_prompt
        
    def inference(self,
                  text):
        """Generate text based on the provided input"""
        
        self.messages = []
        self.messages.append(SystemMessage(content=self.system_prompt))
#         message = self.messages.copy()
        self.messages.append(HumanMessage(content=text))
        res = self.chat(self.messages)
        
        return res.content


class Mistral(GenerativeModel):
    def __init__(self, model_name, model_directory, system_prompt=None, device_assignment='auto', computation_device='cuda'):
        super().__init__(model_name, system_prompt)
        self.model_directory = model_directory
        self.device_assignment = device_assignment
        self.computation_device = computation_device
        self.initialize_model()

    def initialize_model(self):
        """Initialize the model and tokenizer from the provided directory."""
        quantization_configuration = quantization_config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_directory,
            device_map=self.device_assignment,
            quantization_config=quantization_configuration,
            trust_remote_code=True,
        ).to(self.computation_device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_directory,
        )

    def update_system_prompt(self, new_system_prompt):
        """Update the system prompt for generating responses."""
        self.system_prompt = new_system_prompt

    def generate_text(self, input_text, top_p=0.6, temperature=0.8, repetition_penalty=1.0, max_tokens=2000, skip_special_tokens=False, sample=True):
        """Generate a response based on input text and predefined settings."""
        combined_prompt = f"{self.system_prompt}\n{input_text}"

        input_ids = self.tokenizer(combined_prompt, return_tensors="pt").input_ids.to(self.computation_device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                input_ids,
                max_length=max_tokens,
                do_sample=sample,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id
            )

        output_text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens=skip_special_tokens)
        return output_text.strip()

