from __future__ import annotations

import sys

from . import cli, ui


CLI_COMMANDS = {
    "enqueue",
    "worker",
    "status",
    "resume",
    "import-existing",
    "rebuild-english",
    "open-review",
    "open-output",
    "pause",
    "unpause",
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in CLI_COMMANDS:
        return cli.main(args)
    if args and args[0] in {"ui", "gui"}:
        args = args[1:]
    if args:
        return cli.main(args)
    ui.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
