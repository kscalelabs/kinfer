"""Defines the base interface for running model inference.

All kinfer models must implement this interface - the model inputs and outputs
should match the provided schema, and the `__call__` method should take the
inputs and return the outputs according to this schema.
"""

import functools
from abc import ABC, abstractmethod

from kinfer import proto as K


class KModel(ABC):
    """Base interface for running model inference."""

    @abstractmethod
    def get_schema(self) -> K.ModelSchema:
        """Get the model schema."""

    @abstractmethod
    def __call__(self, inputs: K.IO) -> K.IO:
        """Run inference on input data.

        Args:
            inputs: Input data, matching the input schema.

        Returns:
            Model outputs, matching the output schema.
        """

    @functools.cached_property
    def schema(self) -> K.ModelSchema:
        return self.get_schema()

    @property
    def input_schema(self) -> K.IOSchema:
        """Get the input schema."""
        return self.schema.input_schema

    @property
    def output_schema(self) -> K.IOSchema:
        """Get the output schema."""
        return self.schema.output_schema

    @property
    def schema_input_keys(self) -> list[str]:
        """Get all value names from input schemas.

        Returns:
            List of value names from input schema.
        """
        input_names = [value.value_name for value in self.input_schema.values]
        return input_names

    @property
    def schema_output_keys(self) -> list[str]:
        """Get all value names from output schemas.

        Returns:
            List of value names from output schema.
        """
        output_names = [value.value_name for value in self.output_schema.values]
        return output_names
