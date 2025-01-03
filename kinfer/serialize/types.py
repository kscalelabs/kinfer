"""Type conversion utilities for serializers."""

from typing import Type, TypeVar, cast

from kinfer import proto as K

T = TypeVar("T")


def to_value_type(enum_value: T) -> K.Value:
    """Convert an enum value to ValueType."""
    return cast(K.Value, enum_value)


def from_value_type(value_type: K.Value, enum_class: Type[T]) -> T:
    """Convert a ValueType to the specified enum type."""
    return cast(T, value_type)
