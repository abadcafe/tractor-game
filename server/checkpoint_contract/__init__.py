"""Storage contract shared by checkpoint writers and readers."""

CHECKPOINT_OBJECTS_DIR = "objects"
CHECKPOINT_SCHEMA_VERSION = 26
CHECKPOINT_STATE_FILENAME = "state.pt"

__all__ = (
    "CHECKPOINT_OBJECTS_DIR",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_STATE_FILENAME",
)
