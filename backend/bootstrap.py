"""
Administrative Provisioning & Bootstrap Module for DocuMind AI.
Provides explicit, controlled creation and promotion of administrator accounts.
Prevents privilege escalation and account-squatting attacks.
"""

import os
import sys
import argparse
import getpass
from typing import Optional, Tuple
from sqlalchemy.orm import Session

from backend.models import User
from backend.auth import hash_password

RESERVED_ADMIN_USERNAMES = {"admin", "administrator", "root"}


def is_username_reserved(username: str) -> bool:
    """
    Checks whether a username is reserved for system administration.
    Normalizes whitespace and case to ensure strict defense against squatting.
    """
    if not username:
        return False
    normalized = username.strip().lower()
    if normalized in RESERVED_ADMIN_USERNAMES:
        return True
    bootstrap_env_user = os.getenv("DOCUMIND_ADMIN_BOOTSTRAP_USERNAME", "").strip().lower()
    if bootstrap_env_user and normalized == bootstrap_env_user:
        return True
    return False


def provision_admin_account(
    db: Session,
    username: str,
    secret: Optional[str] = None,
    email: Optional[str] = None,
    force_promote: bool = False,
) -> Tuple[bool, str]:
    """
    Explicit, controlled administrative provisioning.

    Rules:
    1. Secret is REQUIRED. If missing or empty, provisioning is refused.
    2. If account does not exist:
       Creates initial admin account with is_admin=True, hashed password, and email.
    3. If account already exists:
       - If user.is_admin is already True: returns (True, message) without modifying.
       - If user.is_admin is False:
         - Automatic/startup calls (force_promote=False): REFUSES promotion due to ownership conflict.
           Existing account is preserved untouched and remains is_admin=False.
         - Explicit operator action (force_promote=True): Elevates is_admin=True and resets password
           to the bootstrap secret.
    """
    clean_username = username.strip()
    if not clean_username:
        return False, "Cannot provision admin: username cannot be empty."

    clean_secret = (secret or "").strip()
    if not clean_secret:
        return False, "Cannot provision admin: bootstrap secret is required and cannot be empty."

    existing_user = db.query(User).filter(User.username == clean_username).first()

    if existing_user:
        if existing_user.is_admin:
            return True, f"Admin account '{existing_user.username}' is already provisioned with administrator privileges."

        # Existing account has is_admin=False
        if not force_promote:
            return False, (
                f"Ownership conflict: An unprivileged account named '{existing_user.username}' already exists. "
                "Automatic elevation is refused to prevent privilege escalation. "
                "Note: Promoting this account via the explicit bootstrap CLI with --force-promote will also reset its password to the bootstrap secret."
            )

        # Explicit operator force promotion with password reset
        try:
            existing_user.is_admin = True
            existing_user.hashed_password = hash_password(clean_secret)
            if email:
                existing_user.email = email.lower().strip()
            db.commit()
            db.refresh(existing_user)
            return True, f"Successfully promoted existing account '{existing_user.username}' to administrator and updated its credentials."
        except Exception as err:
            db.rollback()
            return False, f"Failed to promote account '{existing_user.username}' due to database error: {err.__class__.__name__}"

    # Account does not exist -> safely create new administrator account
    admin_email = (email or f"{clean_username}@documind.local").lower().strip()
    email_conflict = db.query(User).filter(User.email == admin_email).first()
    if email_conflict:
        return False, f"Cannot create admin '{clean_username}': email '{admin_email}' is already associated with another user."

    try:
        new_admin = User(
            username=clean_username,
            email=admin_email,
            hashed_password=hash_password(clean_secret),
            is_active=True,
            is_admin=True,
        )
        db.add(new_admin)
        db.commit()
        db.refresh(new_admin)
        return True, f"Successfully created initial admin account '{clean_username}' with administrator privileges."
    except Exception as err:
        db.rollback()
        return False, f"Failed to create admin account '{clean_username}' due to database error: {err.__class__.__name__}"


def main():
    """CLI entrypoint for explicit operator admin provisioning."""
    parser = argparse.ArgumentParser(
        description="DocuMind AI Administrative Provisioning CLI (Explicit Operator Setup)",
    )
    parser.add_argument(
        "--username",
        help="Admin username to provision or promote (defaults to DOCUMIND_ADMIN_BOOTSTRAP_USERNAME)",
        default=os.getenv("DOCUMIND_ADMIN_BOOTSTRAP_USERNAME", ""),
    )
    parser.add_argument(
        "--secret",
        help=(
            "Bootstrap secret / password for admin account. "
            "WARNING: Supplying secrets on the command line can expose them in shell history and process listings. "
            "If omitted in an interactive terminal, a secure hidden prompt will be used."
        ),
        default=os.getenv("DOCUMIND_ADMIN_BOOTSTRAP_SECRET", ""),
    )
    parser.add_argument(
        "--email",
        help="Admin email address (optional, defaults to <username>@documind.local)",
        default=os.getenv("DOCUMIND_ADMIN_BOOTSTRAP_EMAIL", ""),
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Explicitly promote an existing unprivileged account (Note: This will also reset its password to the bootstrap secret)",
    )

    args = parser.parse_args()

    # Resolve username
    username = args.username.strip()
    if not username:
        if sys.stdin.isatty():
            try:
                username = input("Enter admin username: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                sys.exit(1)
        if not username:
            print("Error: Admin username is required. Pass --username or set DOCUMIND_ADMIN_BOOTSTRAP_USERNAME.")
            sys.exit(1)

    # Resolve secret
    secret = args.secret.strip()
    if not secret:
        if sys.stdin.isatty():
            try:
                secret = getpass.getpass("Enter admin bootstrap secret (hidden): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nOperation cancelled.")
                sys.exit(1)
        if not secret:
            print("Error: Bootstrap secret is required. Omit --secret to enter it via hidden prompt, or set DOCUMIND_ADMIN_BOOTSTRAP_SECRET.")
            sys.exit(1)

    from backend.main import engine
    from sqlalchemy.orm import sessionmaker

    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as db:
        success, msg = provision_admin_account(
            db=db,
            username=username,
            secret=secret,
            email=args.email.strip() or None,
            force_promote=args.force_promote,
        )
        print(f"[DocuMind AI Admin Bootstrap] {msg}")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
