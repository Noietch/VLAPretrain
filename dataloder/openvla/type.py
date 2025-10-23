from abc import ABC, abstractmethod
from typing import Optional, Dict, Optional, Protocol, Union
from PIL.Image import Image

import torch

class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...

class ImageTransform(Protocol):
    def __call__(self, img: Image, **kwargs: str) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...
