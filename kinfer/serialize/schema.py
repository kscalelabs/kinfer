"""Defines utility functions for the schema."""

from kinfer.protos.kinfer_pb2 import Input, InputSchema, Output, OutputSchema, Value, ValueSchema, ValueType


def get_dummy_value(name: str, value_type: ValueType) -> Value:
    value = Value()
    value.value_name = name

    match value_type:
        case ValueType.TENSOR:
            return value
        case _:
            raise ValueError(f"Invalid value type: {value_type}")


def get_dummy_inputs(input_schema: InputSchema) -> Input:
    input_value = Input()
    for value_schema in input_schema.inputs:
        input_value.inputs.append(get_dummy_value(value_schema.name, value_schema.value_types))
    return input_value


def get_dummy_outputs(output_schema: OutputSchema) -> Output:
    output_value = Output()
    for value_schema in output_schema.outputs:
        output_value.outputs.append(get_dummy_value(value_schema.name, value_schema.value_types))
    return output_value
