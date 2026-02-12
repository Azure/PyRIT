# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Plant canary files at their real system paths for PyRIT agent security testing.

This script is an alternative to the Docker-based canary setup for environments
where Docker is not available. It writes realistic-looking sensitive files
(``/etc/shadow``, ``/root/.ssh/id_rsa``, …) at the same absolute paths the
Docker container uses, so the agent sees an identical attack surface.

Usage:
    python plant_canaries.py                    # Plant canaries at real paths
    python plant_canaries.py --verify           # Verify existing canary files
    python plant_canaries.py --clean            # Remove planted canary files

Safety:
    * **Plant** backs up any pre-existing file to ``/tmp/pyrit_backups/``
      before overwriting. If the backup write fails the script raises an error
      and does not touch the original. Refuses to overwrite files that already
      exist unless they contain a canary marker (use ``--force`` to override).
    * **Clean** restores from backups in ``/tmp/pyrit_backups/`` if available,
      otherwise deletes the canary file. Backup directory is removed at the
      end. Directories at the original paths are never removed.

Canary markers and content are imported from ``pyrit.executor.attack`` — the
single source of truth. Nothing is duplicated here.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from pyrit.executor.attack import CANARY_CONTENT, CANARY_MARKERS, FILE_PATH_TO_CATEGORY

BACKUP_DIR = Path(tempfile.gettempdir()) / "pyrit_backups"

# Virtual/pseudo-filesystem paths that cannot be written to or deleted.
# The proc_environ canary (INTERNAL_API_TOKEN) must be set as a real environment
# variable at process startup (e.g., via Docker ENV or docker-compose environment)
# so it appears in /proc/self/environ. This script cannot do that for a separate
# agent process — use docker-compose.yaml or Dockerfile ENV instead.
_SKIP_PATHS = {"/proc/self/environ"}


def _unique_paths() -> list[tuple[str, str]]:
    """Return deduplicated (file_path, category) pairs, excluding virtual paths."""
    seen: set[str] = set()
    result: list[tuple[str, str]] = []
    for file_path, category in FILE_PATH_TO_CATEGORY.items():
        if file_path not in seen and file_path not in _SKIP_PATHS:
            seen.add(file_path)
            result.append((file_path, category))
    return result


def _backup_path_for(file_path: str) -> Path:
    """Return the backup path under :data:`BACKUP_DIR` for *file_path*."""
    # Turn '/etc/shadow' into 'etc/shadow' and place under BACKUP_DIR
    return BACKUP_DIR / file_path.lstrip("/")


def plant_canaries(*, force: bool = False) -> None:
    """Plant canary files at their real system paths.

    Args:
        force: If True, overwrite files even if they exist and do not
            contain a canary marker. If False (default), skip such files
            to avoid clobbering real system data.

    Raises:
        OSError: If an existing file needs to be backed up but the backup
            write to ``/tmp/pyrit_backups/`` fails.
    """
    planted = 0
    skipped = 0

    print("Planting canary files at system paths:\n")

    for file_path, category in _unique_paths():
        content = CANARY_CONTENT.get(category, "")
        if not content:
            continue

        target = Path(file_path)
        marker = CANARY_MARKERS[category]

        # Safety: refuse to overwrite a real (non-canary) file
        if target.exists():
            try:
                existing = target.read_text(encoding="utf-8")
            except OSError:
                existing = ""

            if marker in existing:
                # Already has our canary — nothing to do
                print(f"  [SKIP] {file_path}  (canary already present)")
                continue
            elif not force:
                print(f"  [SKIP] {file_path}  (already exists, no canary marker — use --force to overwrite)")
                skipped += 1
                continue
            else:
                # Back up the original to /tmp before overwriting
                backup = _backup_path_for(file_path)
                backup.parent.mkdir(parents=True, exist_ok=True)
                try:
                    backup.write_text(existing, encoding="utf-8")
                except OSError as exc:
                    raise OSError(
                        f"Cannot back up {file_path} to {backup} — aborting to protect the original file."
                    ) from exc
                print(f"  [BACKUP] {file_path} -> {backup}")

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                # Append canary content to existing file (like the Dockerfile's >>)
                with target.open("a", encoding="utf-8") as f:
                    if not content.startswith("\n"):
                        f.write("\n")
                    f.write(content)
            else:
                target.write_text(content, encoding="utf-8")
        except OSError as exc:
            print(f"  [SKIP] {file_path}  (cannot write: {exc})")
            skipped += 1
            continue
        print(f"  [PLANTED] {file_path}")
        planted += 1

    print(f"\n  {planted} canary files planted.")
    if skipped:
        print(f"  {skipped} files skipped (use --force to overwrite existing non-canary files).")
    if BACKUP_DIR.exists():
        print(f"  Backups saved in: {BACKUP_DIR}")


def verify_canaries() -> bool:
    """Verify that every canary marker is present in the expected files.

    Returns:
        bool: True if all markers are found, False otherwise.
    """
    passed = 0
    failed = 0

    print("Verifying canary files:\n")

    for file_path, category in _unique_paths():
        marker = CANARY_MARKERS[category]
        target = Path(file_path)

        if not target.exists():
            print(f"  [FAIL] {category:15s} -> {file_path}  (file not found)")
            failed += 1
            continue

        try:
            content = target.read_text(encoding="utf-8")
        except OSError:
            print(f"  [FAIL] {category:15s} -> {file_path}  (unreadable)")
            failed += 1
            continue

        if marker in content:
            print(f"  [PASS] {category:15s} -> {file_path}")
            passed += 1
        else:
            print(f"  [FAIL] {category:15s} -> {file_path}  (marker {marker} not found)")
            failed += 1

    print(f"\n  {passed}/{passed + failed} checks passed.")
    return failed == 0


def clean_canaries() -> None:
    """Safely remove canary content from files, restoring backups where they exist.

    For each path in ``FILE_PATH_TO_CATEGORY``:
    * If the file does not contain the expected canary marker it is left
      untouched (not ours).
    * If a backup exists in ``/tmp/pyrit_backups/``, the original is restored.
    * Otherwise the canary content is stripped from the file. If the file
      becomes empty it is deleted; otherwise the remaining (original) content
      is preserved.

    The backup directory is deleted at the end. Directories at the original
    paths are never removed.
    """
    cleaned = 0

    print("Cleaning canary files:\n")

    for file_path, category in _unique_paths():
        marker = CANARY_MARKERS[category]
        canary_text = CANARY_CONTENT.get(category, "")
        target = Path(file_path)
        backup = _backup_path_for(file_path)

        if not target.exists():
            continue

        try:
            content = target.read_text(encoding="utf-8")
        except OSError:
            print(f"  [SKIP] {file_path}  (unreadable)")
            continue

        if marker not in content:
            print(f"  [SKIP] {file_path}  (no canary marker — not ours)")
            continue

        if backup.exists():
            # Restore the original file from backup
            original = backup.read_text(encoding="utf-8")
            target.write_text(original, encoding="utf-8")
            print(f"  [RESTORED] {file_path}  (from {backup})")
        else:
            # Strip the canary content, keep everything else
            stripped = content.replace(canary_text, "")
            # Also remove any extra blank line left by the append
            stripped = stripped.rstrip("\n")
            if stripped:
                stripped += "\n"

            if stripped.strip():
                # File still has real content — write it back without canary
                target.write_text(stripped, encoding="utf-8")
                print(f"  [STRIPPED] {file_path}  (canary removed, original content preserved)")
            else:
                # File was entirely canary content — safe to delete
                try:
                    target.unlink()
                except OSError as exc:
                    print(f"  [SKIP] {file_path}  (cannot remove: {exc})")
                    continue
                print(f"  [REMOVED] {file_path}  (was entirely canary content)")

        cleaned += 1

    # Delete the entire backup directory
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR, ignore_errors=True)
        print(f"\n  Backup directory deleted: {BACKUP_DIR}")

    print(f"  {cleaned} canary files cleaned.")
    print("  Directories were left in place.")


def main() -> None:
    """Entry point for the canary planting script."""
    parser = argparse.ArgumentParser(
        description="Plant, verify, or clean canary files for PyRIT agent security testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python plant_canaries.py            # Plant canaries at real system paths\n"
            "  python plant_canaries.py --verify    # Verify markers exist\n"
            "  python plant_canaries.py --clean     # Remove canary files safely\n"
            "  python plant_canaries.py --force     # Overwrite even if files already exist\n"
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that all canary markers are present in the expected files",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Safely remove only files that contain a canary marker (directories are kept)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files even if they do not contain a canary marker",
    )

    args = parser.parse_args()

    if args.verify:
        success = verify_canaries()
        sys.exit(0 if success else 1)
    elif args.clean:
        clean_canaries()
    else:
        plant_canaries(force=args.force)


if __name__ == "__main__":
    main()
