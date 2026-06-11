"""Attribute-dict builders for the bundle (the only custom format surface)."""


def root_attrs() -> dict[str, object]:
    """Return the custom attributes for the bundle's root group.

    The root group carries only Zarr's own consolidated_metadata; the format
    defines no custom root attributes, so this is always empty.
    """
    raise NotImplementedError
