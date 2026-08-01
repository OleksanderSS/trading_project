"""DISARMED. Do not run this script.

It walked every .py file under src/ and replaced

    except Exception as e:
with
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:

writing each file back in place. That single pass is the origin of the ~653
identical handlers in this codebase.

The tuple reads as exhaustive and is not. It omits OSError, RuntimeError,
IndexError and every third-party exception. Three defects found during the
2026-07/08 audit trace directly to it:

  - CatBoostError (inherits from Exception) escaped a handler and took down
    a whole pipeline stage;
  - sqlite3.IntegrityError escaped and silently lost tracked events;
  - yaml.YAMLError escaped FileManager.load_yaml, so a malformed config
    raised a raw ParserError while malformed JSON -- whose JSONDecodeError
    happens to be a ValueError -- was handled correctly. Two loaders in one
    class disagreeing about the same class of failure.

Re-running this would undo every broadening fix made since, in one pass:
ErrorHandler.safe_execute, ErrorHandler.graceful_degradation and
FileManager.load_yaml/load_json all deliberately catch broadly now.

Narrowing an exception handler is a decision about one specific call, based
on what that call can actually raise. It cannot be made by regex across a
repository. If handlers need review, use
tests/contracts/test_silent_failure_paths.py, which counts them and holds
the count from rising.

The original body is kept below, unreachable, as the record of what ran.
"""

raise SystemExit(__doc__)


# ---------------------------------------------------------------------------
# Original implementation -- retained for the record, never executed.
# ---------------------------------------------------------------------------
#
# import os
# import re
#
# def refactor_broad_exceptions(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         content = f.read()
#
#     pattern = r'(\s*)except Exception as e:'
#     replacement = (
#         r'\1except (ValueError, TypeError, AttributeError, KeyError, '
#         r'ZeroDivisionError) as e:'
#     )
#     new_content = re.sub(pattern, replacement, content)
#
#     if new_content != content:
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write(new_content)
#         print(f"Refactored: {file_path}")
#         return True
#     return False
#
# for root, _, files in os.walk('src'):
#     for file in files:
#         if file.endswith('.py'):
#             refactor_broad_exceptions(os.path.join(root, file))
