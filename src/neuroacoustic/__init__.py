"""NeuroAcoustic Engine: Real-time generative cognitive audio."""

__version__ = "0.1.0"

from .synthesis.dsp_engine import CognitiveAudioSynthesizer
from .translation.semantic_mapper import SemanticMapper

__all__ = ["CognitiveAudioSynthesizer", "SemanticMapper"]
