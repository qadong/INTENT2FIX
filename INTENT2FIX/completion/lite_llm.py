
from typing import List, Dict, Any

import litellm

from INTENT2FIX.completion.model import Completion


class LiteLLM:
    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs

    def generate(self, prompt: str) -> "LiteLLM":
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = litellm.completion(
                model=self.model,
                messages=messages,
                **self.kwargs
            )
            
            self._last_message = response.choices[0].message.content
            # To be compatible with the loop, I will return self.
            return self
            
        except Exception as e:
            # Handle exceptions appropriately
            print(f"Error during completion: {e}")
            raise

    @property
    def message(self):
        if not hasattr(self, '_last_message'):
            return ""
        return self._last_message

