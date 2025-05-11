"""Defines common utilities for exporting models."""


def get_shape(
    name: str,
    num_joints: int | None = None,
    num_commands: int | None = None,
    carry_shape: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    match name:
        case "joint_angles":
            if num_joints is None:
                raise ValueError("`num_joints` must be provided when using `joint_angles`")
            return (num_joints,)

        case "joint_angular_velocities":
            if num_joints is None:
                raise ValueError("`num_joints` must be provided when using `joint_angular_velocities`")
            return (num_joints,)

        case "projected_gravity":
            return (3,)

        case "accelerometer":
            return (3,)

        case "gyroscope":
            return (3,)

        case "command":
            if num_commands is None:
                raise ValueError("`num_commands` must be provided when using `command`")
            return (num_commands,)

        case "carry":
            if carry_shape is None:
                raise ValueError("`carry_shape` must be provided for `carry`")
            return carry_shape

        case _:
            raise ValueError(f"Unknown tensor name: {name}")
