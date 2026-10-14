"""Spiking modules with explicit physical-time state semantics."""

from .task_routed_memory import TaskRoutedSpikingMemory
from .prompt_dynamics import TaskRelationalSpikingPrompt

__all__ = ["TaskRoutedSpikingMemory", "TaskRelationalSpikingPrompt"]
