"""
Unit tests for triage_engine/security.py: CSRF token handling, the sliding-
window rate limiter, and transparent legacy-plaintext-password migration.
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from triage_engine import security


class FakeSession(dict):
    """Minimal stand-in for Flask's session object (a dict works fine for
    what get_or_create_csrf_token/validate_csrf actually touch)."""
    pass


class TestCsrf(unittest.TestCase):
    def test_token_is_created_and_persisted(self):
        sess = FakeSession()
        token1 = security.get_or_create_csrf_token(sess)
        token2 = security.get_or_create_csrf_token(sess)
        self.assertEqual(token1, token2)
        self.assertGreater(len(token1), 20)

    def test_valid_token_passes(self):
        sess = FakeSession()
        token = security.get_or_create_csrf_token(sess)
        self.assertTrue(security.validate_csrf(sess, token))

    def test_missing_or_wrong_token_fails(self):
        sess = FakeSession()
        security.get_or_create_csrf_token(sess)
        self.assertFalse(security.validate_csrf(sess, None))
        self.assertFalse(security.validate_csrf(sess, ""))
        self.assertFalse(security.validate_csrf(sess, "wrong-token"))

    def test_no_token_in_session_fails_closed(self):
        sess = FakeSession()
        self.assertFalse(security.validate_csrf(sess, "anything"))


class TestRateLimiter(unittest.TestCase):
    def test_allows_up_to_max_then_blocks(self):
        rl = security.RateLimiter(max_attempts=3, window_seconds=60)
        self.assertTrue(rl.allow("k"))
        self.assertTrue(rl.allow("k"))
        self.assertTrue(rl.allow("k"))
        self.assertFalse(rl.allow("k"))

    def test_different_keys_are_independent(self):
        rl = security.RateLimiter(max_attempts=1, window_seconds=60)
        self.assertTrue(rl.allow("a"))
        self.assertTrue(rl.allow("b"))
        self.assertFalse(rl.allow("a"))

    def test_window_expiry_allows_again(self):
        rl = security.RateLimiter(max_attempts=1, window_seconds=0.05)
        self.assertTrue(rl.allow("k"))
        self.assertFalse(rl.allow("k"))
        time.sleep(0.08)
        self.assertTrue(rl.allow("k"))

    def test_parse_rate_string(self):
        self.assertEqual(security.parse_rate_string("10 per minute"), (10, 60))
        self.assertEqual(security.parse_rate_string("5 per second"), (5, 1))
        self.assertEqual(security.parse_rate_string("2 per hour"), (2, 3600))
        self.assertEqual(security.parse_rate_string("garbage input"), (10, 60))


class FakeUserRow(dict):
    def __getitem__(self, key):
        return dict.__getitem__(self, key)


class FakeConn:
    def __init__(self):
        self.executed = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def commit(self):
        pass


class TestPasswordMigration(unittest.TestCase):
    def test_hashed_password_verifies_normally(self):
        conn = FakeConn()
        user = FakeUserRow(id=1, password=security.hash_password("demo123"))
        self.assertTrue(security.verify_and_upgrade_password(conn, user, "demo123"))
        self.assertFalse(security.verify_and_upgrade_password(conn, user, "wrong"))
        self.assertEqual(conn.executed, [])  # no migration needed, no UPDATE issued

    def test_legacy_plaintext_verifies_and_triggers_upgrade(self):
        conn = FakeConn()
        user = FakeUserRow(id=42, password="demo123")  # legacy plaintext row
        self.assertTrue(security.verify_and_upgrade_password(conn, user, "demo123"))
        self.assertEqual(len(conn.executed), 1)
        sql, params = conn.executed[0]
        self.assertIn("UPDATE users SET password", sql)
        self.assertEqual(params[1], 42)
        self.assertTrue(security.is_hashed_password(params[0]))

    def test_legacy_plaintext_wrong_password_fails_without_upgrade(self):
        conn = FakeConn()
        user = FakeUserRow(id=1, password="demo123")
        self.assertFalse(security.verify_and_upgrade_password(conn, user, "wrong"))
        self.assertEqual(conn.executed, [])

    def test_missing_user_row_fails_closed(self):
        conn = FakeConn()
        self.assertFalse(security.verify_and_upgrade_password(conn, None, "anything"))


if __name__ == "__main__":
    unittest.main()
