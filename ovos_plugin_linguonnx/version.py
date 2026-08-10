# Version block follows OVOS convention: plain module-level integers between
# the marker comments, so the release automation can rewrite them and code can
# read them without importing a build backend.
# START_VERSION_BLOCK
VERSION_MAJOR = 0
VERSION_MINOR = 3
VERSION_BUILD = 2
VERSION_ALPHA = 1
# END_VERSION_BLOCK

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_BUILD}"
if VERSION_ALPHA:
    __version__ += f"a{VERSION_ALPHA}"
