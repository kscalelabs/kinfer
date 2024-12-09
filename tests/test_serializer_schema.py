"""Tests the schema serializer."""

from kinfer.protos.kinfer_pb2 import InputSchema, OutputSchema, ValueType, ValueSchema, Output
from kinfer.serialize.schema import get_dummy_inputs, get_dummy_outputs


def test_serialize_schema() -> None:
    input_schema = InputSchema(
        inputs=[
            ValueSchema(name="input_1", value_types=value_type)
            for value_type in ValueType.values()
        ]
    )

    dummy_input = get_dummy_inputs(input_schema)

    breakpoint()

    asdf
