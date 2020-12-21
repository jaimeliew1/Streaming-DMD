"""
DEBUGGING FUNCTIONS These commands (ic, profile) are convenient for code
introspection during development. They should be removed when the repository is
published.
"""
import icecream
import profilehooks

builtins = __import__("builtins")
setattr(builtins, "ic", icecream.ic)
setattr(builtins, "profile", profilehooks.profile)
