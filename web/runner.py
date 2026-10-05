"""
The sandbox child. Reads {"code", "df"} as a pickle on stdin, writes the
result as a pickle on stdout, and nothing else goes to stdout -- the parent
parses it.

Run as: python -m web.runner
"""

import os
import pickle
import sys


def _apply_limits():
    """
    Caps so a runaway submission takes itself down instead of the host. Not
    available on every platform, so failures here are non-fatal.
    """
    try:
        import resource
    except ImportError:
        return

    mb = int(os.environ.get("DRILL_EXEC_MEMORY_MB") or 2048)

    for name, limit in (
        ("RLIMIT_AS", mb * 1024 * 1024),
        ("RLIMIT_CPU", int(float(os.environ.get("DRILL_EXEC_TIMEOUT") or 15)) + 5),
        ("RLIMIT_NPROC", 64),
        ("RLIMIT_CORE", 0),
    ):
        try:
            soft, hard = resource.getrlimit(getattr(resource, name))
            ceiling = limit if hard in (resource.RLIM_INFINITY,) else min(limit, hard)
            resource.setrlimit(getattr(resource, name), (ceiling, hard))
        except (AttributeError, ValueError, OSError):
            continue


def main():
    _apply_limits()

    request = pickle.loads(sys.stdin.buffer.read())

    # Imported after the limits are in place, and after stdin is consumed.
    from .executor import run_inprocess

    result = run_inprocess(request["code"], request["df"])

    answer = result.answer

    try:
        pickle.dumps(answer, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # Something unpicklable -- an open file, a lambda. The verdict needs
        # the real object, so say so rather than silently sending None.
        answer = None
        result.answer_bound = False
        result.traceback = (
            "`answer` holds something that cannot leave the sandbox "
            "(a file handle, a lambda, a generator). Return a DataFrame, "
            "Series, or scalar."
        )
        result.ok = False

    value, has_value = result.value, result.has_value

    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        # An echoed value is a convenience, never the verdict -- drop it
        # rather than failing the run over something unpicklable.
        value, has_value = None, False

    payload = {
        "ok": result.ok,
        "stdout": result.stdout,
        "traceback": result.traceback,
        "answer": answer,
        "answer_bound": result.answer_bound,
        "value": value,
        "has_value": has_value,
        "names": result.names,
    }

    sys.stdout.buffer.write(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
