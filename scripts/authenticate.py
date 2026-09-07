"""Sign in through Kaggle's official browser approval flow on a remote terminal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kagglesdk import KaggleClient, KaggleCredentials, KaggleOAuth

from nfl_trajectory.runtime import Run


def authenticate(root: Path, *, force: bool = False) -> int:
    """Reuse valid OAuth credentials or ask Kaggle to authorize this persistent space."""
    try:
        with Run(root, "authenticate") as run, KaggleClient() as client:
            credentials = KaggleCredentials.load(client=client)
            if credentials is not None and not force:
                # The SDK refreshes expired access tokens and persists them outside this project.
                credentials.get_access_token()
                credentials.introspect()
                run.event("authentication_reused", method="oauth")
                print("Saved Kaggle sign-in verified. You can continue with nfl download.")
                return 0

            print(
                "Open the Kaggle link below, sign in, and approve access.\n"
                "Paste the one-time verification code from that page into this terminal.\n"
                "The approval link and code are not written to the project logs.",
                flush=True,
            )
            # Match the official CLI's scopes and remote-terminal OAuth flow.
            # The SDK manages PKCE, code exchange, credential validation, and saving.
            KaggleOAuth(client=client).authenticate(
                scopes=["resources.admin:*"], no_launch_browser=True
            )
            run.event("authentication_verified", method="oauth")
        return 0
    except (EOFError, KeyboardInterrupt):
        print("Sign-in cancelled. Run this same command when you are ready.")
        return 130
    except Exception as exc:
        # SDK exception messages may contain credentials or session-specific URLs.
        print(
            f"Sign-in did not complete ({type(exc).__name__}). "
            "Retry this command; use --force if you need a new browser approval."
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Request a new browser approval.")
    args = parser.parse_args(argv)
    return authenticate(Path(__file__).resolve().parents[1], force=args.force)


if __name__ == "__main__":
    sys.exit(main())
