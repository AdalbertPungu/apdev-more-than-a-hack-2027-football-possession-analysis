import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import validate_submission


class ValidateSubmissionTests(unittest.TestCase):
    def validate_text(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "submission.csv")
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(text)
            with patch.object(validate_submission, "N_FRAMES", 3):
                return validate_submission.validate(path)

    def test_valid_file(self):
        self.assertEqual(
            self.validate_text("frame,label\n0,H\n1, a \n2,X\n"),
            {"A": 1, "D": 0, "H": 1, "X": 1},
        )

    def test_bad_frame_order_and_non_integer(self):
        for value in ("2", "0_0", "0.0"):
            with self.subTest(frame=value):
                result = self.validate_text("frame,label\n{},H\n1,A\n2,X\n".format(value))
                self.assertIsInstance(result, str)

    def test_bad_header_labels_and_row_count(self):
        for text in (
            "frame,label,extra\n0,H,1\n1,A,1\n2,X,1\n",
            "frame,label\n0,H\n1,\n2,X\n",
            "frame,label\n0,H\n1,Z\n2,X\n",
            "frame,label\n0,H\n1,A\n",
        ):
            with self.subTest(text=text):
                self.assertIsInstance(self.validate_text(text), str)


if __name__ == "__main__":
    unittest.main()
