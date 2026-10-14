"""Real ASGI authentication: no public chat or controller access after opt-in."""
import os

from fastapi.testclient import TestClient

from blackthorn.app import create_app
from blackthorn.auth import hash_password
from blackthorn.config import Settings


def test_login_reset_revokes_and_csrf(monkeypatch, tmp_path):
    monkeypatch.setenv('BLACKTHORN_AUTH_REQUIRED', '1')
    monkeypatch.setenv('BLACKTHORN_AUTH_SIGNING_KEY', 'x' * 48)
    monkeypatch.setenv('BLACKTHORN_AUTH_PASSWORD_HASH', hash_password('first-password-123456789'))
    with TestClient(create_app(Settings(store='sqlite', sqlite_path=str(tmp_path / 'auth.sqlite')), gpu_daemon=False)) as client:
        assert client.get('/api/chat/sessions').status_code == 401
        assert client.get('/api/version').status_code == 401
        assert client.get('/api/auth/status').json()['authenticated'] is False
        assert client.post('/api/auth/login', json={'password': 'first-password-123456789'}, headers={'Origin':'https://evil.example'}).status_code == 403
        assert client.post('/api/auth/login', json={'password': 'first-password-123456789'}, headers={'Origin':'http://testserver'}).status_code == 200
        old_cookie = client.cookies.get('bt_owner')
        assert client.get('/api/chat/sessions').status_code == 200
        assert client.post('/api/auth/reset',json={'current_password':'first-password-123456789','new_password':'a-different-password-123456789'},headers={'Origin':'http://testserver'}).status_code == 200
        assert client.get('/api/chat/sessions').status_code == 200
        client.cookies.clear()
        client.cookies.set('bt_owner', old_cookie)
        assert client.get('/api/chat/sessions').status_code == 401
        assert client.post('/api/auth/login', json={'password':'a-different-password-123456789'}, headers={'Origin':'http://testserver'}).status_code == 200
        assert client.post('/api/auth/logout', headers={'Origin':'http://testserver'}).status_code == 200
        assert client.get('/api/chat/sessions').status_code == 401
