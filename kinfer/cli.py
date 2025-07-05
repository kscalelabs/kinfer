"""Command-line interface for kinfer model inspection and packaging utility."""

import tarfile
from pathlib import Path

import click

from kinfer.commands import CommandTypeRegistry
from kinfer.commands_registry import DeploymentCommandRegistry
from kinfer.rust_bindings import metadata_from_json


@click.group()
@click.version_option()
def cli():
    """Kinfer model inspection and packaging utility."""
    pass


@cli.command()
@click.argument('command_name', required=False)
@click.option('--details', is_flag=True, help='Show detailed field information for the specified command')
def commands(command_name, details):
    """List all available command types, or show info for a specific command."""
    
    if command_name:
        # Show specific command info (existing behavior)
        cmd_def = DeploymentCommandRegistry.get_command_definition(command_name)
        if not cmd_def:
            click.echo(f"Error: Unknown command type '{command_name}'", err=True)
            click.echo(f"Available commands: {', '.join(DeploymentCommandRegistry.list_command_types())}")
            raise click.Abort()
        
        if details:
            # Show detailed view for specific command
            click.echo(f"Command Details: {command_name}")
            click.echo("=" * (len(command_name) + 17))
            
            click.echo(f"\nDescription:")
            click.echo(f"  {cmd_def.payload_schema.description}")
            
            click.echo(f"\nTransport Envelope:")
            click.echo(f"  Payload Type: {cmd_def.transport_envelope.payload_type.value}")
            click.echo(f"  Payload Length: {cmd_def.transport_envelope.payload_length}")
            click.echo(f"  Codec Info: {cmd_def.transport_envelope.codec_info}")
            
            if cmd_def.payload_schema.fields:
                click.echo(f"\nFields ({len(cmd_def.payload_schema.fields)}):")
                for field in cmd_def.payload_schema.fields:
                    units_str = f" [{field.units}]" if field.units else ""
                    range_str = f" (range: {field.range})" if field.range else ""
                    click.echo(f"  • {field.name}{units_str}: {field.description}{range_str}")
            else:
                click.echo(f"\nFields: None")
            
            if cmd_def.payload_schema.custom_properties:
                click.echo(f"\nCustom Properties:")
                for key, value in cmd_def.payload_schema.custom_properties.items():
                    if isinstance(value, dict):
                        click.echo(f"  {key}: {type(value).__name__} (complex)")
                    elif isinstance(value, list):
                        click.echo(f"  {key}: {len(value)} items")
                    else:
                        click.echo(f"  {key}: {value}")
            
            # Show decoder info
            decoder_info = DeploymentCommandRegistry.get_decoder_info(command_name)
            if decoder_info:
                click.echo(f"\nDecoder Instructions:")
                click.echo(f"  {decoder_info['decoder_instructions']}")
            
            click.echo(f"\nKinfer Version: {cmd_def.kinfer_version}")
        else:
            # Show basic view for specific command
            click.echo(f"Command: {command_name}")
            click.echo("=" * (len(command_name) + 9))
            
            click.echo(f"\nDescription:")
            # Show first sentence only
            first_sentence = cmd_def.payload_schema.description.split('.')[0] + '.'
            click.echo(f"  {first_sentence}")
            
            click.echo(f"\nPayload Type: {cmd_def.transport_envelope.payload_type.value}")
            
            if cmd_def.payload_schema.fields:
                click.echo(f"Fields: {len(cmd_def.payload_schema.fields)}")
                for field in cmd_def.payload_schema.fields:
                    units_str = f" [{field.units}]" if field.units else ""
                    click.echo(f"  • {field.name}{units_str}")
            else:
                click.echo(f"Fields: None")
            
            # Add yellow note about details flag
            click.echo()
            click.echo(click.style("💡 Use --details for full implementation details:", fg='yellow'))
            click.echo(click.style(f"   kinfer commands {command_name} --details", fg='yellow'))
    else:
        # Show simplified summary of all commands
        click.echo("Available Command Types:")
        click.echo("=" * 50)
        
        command_types = DeploymentCommandRegistry.get_all_command_types()
        
        # Find the longest command name for consistent formatting
        max_name_len = max(len(name) for name in command_types.keys())
        
        for command_type in sorted(command_types.keys()):
            cmd_def = DeploymentCommandRegistry.get_command_definition(command_type)
            if cmd_def:
                # Get first sentence of description
                first_sentence = cmd_def.payload_schema.description.split('.')[0] + '.'
                payload_type = cmd_def.transport_envelope.payload_type.value
                
                # Format: command_name (padded) | payload_type | description
                name_padded = command_type.ljust(max_name_len)
                payload_padded = payload_type.ljust(12)  # Consistent payload type width
                click.echo(f"  {name_padded} | {payload_padded} | {first_sentence}")
        
        # Add yellow note about specific command info
        click.echo()
        click.echo(click.style("💡 Show info for a specific command:", fg='yellow'))
        click.echo(click.style("   kinfer commands <command_name>", fg='yellow'))
        click.echo(click.style("   kinfer commands <command_name> --details", fg='yellow'))
        

@cli.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
def inspect(model_path: Path):
    """Inspect a kinfer model file and display metadata."""
    
    if not model_path.suffix == '.kinfer':
        click.echo(f"Warning: File '{model_path}' doesn't have .kinfer extension", err=True)
    
    try:
        # Extract metadata from the kinfer file
        with tarfile.open(model_path, 'r:gz') as tar:
            metadata_file = tar.extractfile('metadata.json')
            if metadata_file is None:
                click.echo("Error: No metadata.json found in model file", err=True)
                raise click.Abort()
            
            metadata_json = metadata_file.read().decode('utf-8')
            metadata = metadata_from_json(metadata_json)
        
        # Print comprehensive model information
        click.echo(f"Kinfer Model Inspection: {model_path.name}")
        click.echo("=" * 60)
        
        click.echo(f"\nGeneral Information:")
        click.echo(f"  Model Path: {model_path.absolute()}")
        click.echo(f"  Kinfer Version: {metadata.kinfer_version or 'Unknown'}")
        click.echo(f"  File Size: {model_path.stat().st_size / (1024*1024):.1f} MB")
        
        click.echo(f"\nModel Configuration:")
        click.echo(f"  Joint Count: {len(metadata.joint_names)}")
        click.echo(f"  Commands: {metadata.num_commands or 'Unknown'}")
        click.echo(f"  Carry Size: {metadata.carry_size}")
        
        click.echo(f"\nJoint Names:")
        for i, joint_name in enumerate(metadata.joint_names, 1):
            click.echo(f"  {i:2d}. {joint_name}")
        
        if metadata.joint_biases:
            click.echo(f"\nJoint Biases:")
            for bias in metadata.joint_biases:
                click.echo(f"  {bias.joint_name}: ref={bias.reference_angle:.3f}, weight={bias.weight:.3f}")
        else:
            click.echo(f"\nJoint Biases: None")
        
        if metadata.command_type_info:
            info = metadata.command_type_info
            click.echo(f"\nCommand Type: {info.command_type}")
            click.echo(f"Description: {info.description.strip()}")
            
            if info.fields:
                click.echo(f"Command Fields:")
                for field in info.fields:
                    units_str = f" [{field.units}]" if field.units else ""
                    range_str = f" (range: {field.range})" if field.range else ""
                    click.echo(f"  • {field.name}{units_str}: {field.description}{range_str}")
            else:
                click.echo(f"Command Fields: None")
        else:
            click.echo(f"\nCommand Type: Not specified")
        
        # Validate command compatibility
        if metadata.command_type_info:
            command_type = metadata.command_type_info.command_type
            if DeploymentCommandRegistry.validate_command_type(command_type):
                click.echo(f"\n✓ Command type '{command_type}' is registered in the registry")
            else:
                click.echo(f"\n⚠ Command type '{command_type}' is NOT registered in the registry")
        
        click.echo(f"\nInspection completed successfully!")
        
    except Exception as e:
        click.echo(f"Error inspecting model: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.argument('model_path', type=click.Path(exists=True, path_type=Path))
def envelope(model_path: Path):
    """Inspect transport envelope of a kinfer model."""
    try:
        with tarfile.open(model_path, 'r:gz') as tar:
            metadata_file = tar.extractfile('metadata.json')
            if metadata_file is None:
                click.echo("Error: No metadata.json found in model file", err=True)
                raise click.Abort()
            
            metadata_json = metadata_file.read().decode('utf-8')
            metadata = metadata_from_json(metadata_json)
        
        click.echo(f"Transport Envelope: {model_path.name}")
        click.echo("=" * 50)
        
        if metadata.command_type_info:
            command_type = metadata.command_type_info.command_type
            envelope = DeploymentCommandRegistry.get_transport_envelope(command_type)
            
            if envelope:
                click.echo(f"Command Type: {command_type}")
                click.echo(f"Payload Type: {envelope.payload_type.value}")
                click.echo(f"Payload Length: {envelope.payload_length}")
                click.echo(f"Codec Info: {envelope.codec_info}")
                
                # Show decoder info for deployment
                decoder_info = DeploymentCommandRegistry.get_decoder_info(command_type)
                if decoder_info:
                    click.echo(f"\nDecoder Instructions:")
                    click.echo(f"  {decoder_info['decoder_instructions']}")
            else:
                click.echo(f"⚠ Unknown command type '{command_type}' - not in registry")
                click.echo(f"Available command types: {DeploymentCommandRegistry.list_command_types()}")
        else:
            click.echo("No transport envelope information available")
            
    except Exception as e:
        click.echo(f"Error inspecting envelope: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()