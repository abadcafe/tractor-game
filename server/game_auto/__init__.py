"""Public pure Auto strategy shared by runtime play and training."""

from ._decision import AutoCommand, choose_auto_command

__all__ = ("AutoCommand", "choose_auto_command")
