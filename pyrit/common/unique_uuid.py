# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid


def generate_unique_uuid(*, existing_uuids: list[str | uuid.UUID]) -> uuid.UUID:
    """Generate a new string UUID-4 avoiding collisions

    Args:
        existing_uuids: Existing UUIDs or strings to be avoided
    Returns:
        A new UUID-4 string.
    """
    existing_uuids = [str(uid) for uid in existing_uuids]
    new_uuid = uuid.uuid4()
    while str(new_uuid) in existing_uuids:
        new_uuid = uuid.uuid4()
    return new_uuid
