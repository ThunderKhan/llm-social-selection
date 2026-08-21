class StorageError(Exception):
    """Base error for persistent experimental records."""


class SchemaVersionError(StorageError):
    """The database schema version is unsupported."""


class AlreadyCommittedError(StorageError):
    """A completed round already occupies the requested trial boundary."""


class IntegrityError(StorageError):
    """Stored data or a requested write violates research-record integrity."""


class NotFoundError(StorageError):
    """A requested persistent record does not exist."""
