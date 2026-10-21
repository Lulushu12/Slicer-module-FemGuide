"""Run the FemGuide wizard self-test inside Slicer and fail on any error.

Qt swallows exceptions raised in event callbacks (they are only printed), so
a test can "pass" while a callback crashed. This runner records everything
written to stderr and fails if a traceback appears anywhere.

    Slicer --no-splash --no-main-window \\
        --additional-module-paths modules/femguide \\
        --python-script scripts/slicer_selftest.py

Exit code 0 means the test passed cleanly.
"""

import io
import sys
import traceback

import slicer


class _Tee(io.TextIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.seen = []

    def write(self, text):
        self.seen.append(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


tee = _Tee(sys.stderr)
sys.stderr = tee
passed = False
try:
    import FemGuide

    FemGuide.FemGuideTest().runTest()
    passed = True
except Exception:
    traceback.print_exc()
finally:
    sys.stderr = tee.stream

crashed_in_callback = any("Traceback" in chunk for chunk in tee.seen)
if passed and not crashed_in_callback:
    print("FEMGUIDE SELFTEST: PASSED")
    slicer.app.exit(0)
else:
    print("FEMGUIDE SELFTEST: FAILED" + (" (error in a callback)" if passed else ""))
    slicer.app.exit(1)
