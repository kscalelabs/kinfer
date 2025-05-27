"""Runs common tests on K-Infer components."""

import json

from kinfer.rust_bindings import PyInputType, PyModelMetadata, metadata_from_json


def test_model_metadata_serialization() -> None:
    metadata = PyModelMetadata(
        joint_names=["joint1", "joint2"],
        num_commands=2,
        carry_size=[1],
    )

    # Checks serialization and deserialization.
    metadata_json = metadata.to_json()
    assert metadata_from_json(metadata_json) == metadata

    # Checks equality.
    metadata_copy = json.loads(metadata_json)
    metadata_copy["carry_size"] = [2]
    assert metadata_from_json(json.dumps(metadata_copy)) != metadata


def test_input_type_serialization() -> None:
    metadata = PyModelMetadata(
        joint_names=["joint1", "joint2"],
        num_commands=2,
        carry_size=[1],
    )

    # Tests joint angles.
    input_type = PyInputType(input_type="joint_angles")
    assert input_type.get_shape(metadata) == [2]

    # Tests joint angular velocities.
    input_type = PyInputType(input_type="joint_angular_velocities")
    assert input_type.get_shape(metadata) == [2]

    # Tests projected gravity.
    input_type = PyInputType(input_type="projected_gravity")
    assert input_type.get_shape(metadata) == [3]

    # Tests accelerometer.
    input_type = PyInputType(input_type="accelerometer")
    assert input_type.get_shape(metadata) == [3]

    # Tests gyroscope.
    input_type = PyInputType(input_type="gyroscope")
    assert input_type.get_shape(metadata) == [3]

    # Tests command.
    input_type = PyInputType(input_type="command")
    assert input_type.get_shape(metadata) == [2]

    # Tests time.
    input_type = PyInputType(input_type="time")
    assert input_type.get_shape(metadata) == [1]
