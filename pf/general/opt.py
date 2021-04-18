from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar('T')

@dataclass(frozen=True)
class Opt(Generic[T]):
    value: Optional[T]

    def or_else(self, fallback: T) -> T:
        if self.value is None:
            return fallback
        return self.value
