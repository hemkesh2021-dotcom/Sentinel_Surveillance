import base64
import unittest
from unittest.mock import patch
import dashboard

class DashboardTests(unittest.TestCase):
    def setUp(self):
        dashboard.DASH_PASS = 'test-pass'
        dashboard._auth_failures.clear()
        self.client = dashboard.app.test_client()
        token = base64.b64encode(b'admin:test-pass').decode()
        self.headers = {'Authorization': 'Basic ' + token}

    def test_all_routes_require_auth(self):
        for route in ['/', '/stream', '/status', '/log']:
            self.assertEqual(self.client.get(route).status_code, 401)
        self.assertEqual(self.client.post('/chat', json={'message': 'hi'}).status_code, 401)

    def test_invalid_chat_never_calls_model(self):
        with patch.object(dashboard, 'ask_ai') as model:
            for payload in [[], {}, {'message': 1}, {'message': ' '}, {'message': 'x' * 2001}]:
                self.assertEqual(self.client.post('/chat', json=payload, headers=self.headers).status_code, 400)
            self.assertEqual(self.client.post('/chat', json={'message': 'x' * 9000}, headers=self.headers).status_code, 413)
            model.assert_not_called()

    def test_chat_capacity_and_release(self):
        dashboard._chat_slot.acquire()
        try:
            self.assertEqual(self.client.post('/chat', json={'message': 'hi'}, headers=self.headers).status_code, 429)
        finally:
            dashboard._chat_slot.release()
        with patch.object(dashboard, 'ask_ai', return_value='answer'):
            self.assertEqual(self.client.post('/chat', json={'message': 'hi'}, headers=self.headers).json['response'], 'answer')

    def test_auth_throttle(self):
        for _ in range(10):
            self.assertEqual(self.client.get('/status').status_code, 401)
        self.assertEqual(self.client.get('/status').status_code, 429)
