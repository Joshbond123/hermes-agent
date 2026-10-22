"""Real ASGI authentication: no public chat or controller access after opt-in."""

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


import pytest
from blackthorn.app import build_services
from blackthorn.tests.fakes import ServerThread


@pytest.mark.asyncio
async def test_browser_login_remembers_and_settings_reset(monkeypatch, tmp_path, browser):
    monkeypatch.setenv('BLACKTHORN_AUTH_REQUIRED', '1')
    monkeypatch.setenv('BLACKTHORN_AUTH_SIGNING_KEY', 'z' * 48)
    monkeypatch.setenv('BLACKTHORN_AUTH_PASSWORD_HASH', hash_password('owner-first-passphrase-123'))
    settings = Settings(store='sqlite', sqlite_path=str(tmp_path / 'browser-auth.sqlite'))
    services = build_services(settings)
    server = ServerThread(create_app(settings, services=services, gpu_daemon=False)).start()
    ctx = await browser.new_context()
    try:
        page = await ctx.new_page()
        await page.goto(server.url)
        await page.get_by_test_id('login-page').wait_for()
        await page.get_by_label('Password', exact=True).fill('owner-first-passphrase-123')
        await page.get_by_role('button', name='Sign in').click()
        await page.get_by_test_id('composer-input').wait_for(timeout=10000)
        await page.reload()
        await page.get_by_test_id('composer-input').wait_for(timeout=10000)
        await page.get_by_test_id('settings-open').click()
        await page.get_by_label('Current password').fill('owner-first-passphrase-123')
        await page.get_by_label('New password (16+ characters)').fill('new-owner-passphrase-123')
        await page.get_by_label('Confirm new password').fill('new-owner-passphrase-123')
        await page.get_by_role('button', name='Change password').click()
        await page.get_by_text('Password changed. Other sessions have been signed out.').wait_for(timeout=10000)
    finally:
        await ctx.close()
        server.stop()


def test_public_source_password_cannot_bypass_configured_hash(monkeypatch, tmp_path):
    monkeypatch.setenv('BLACKTHORN_AUTH_REQUIRED', '1')
    monkeypatch.setenv('BLACKTHORN_AUTH_SIGNING_KEY', 'z' * 48)
    monkeypatch.setenv('BLACKTHORN_AUTH_PASSWORD_HASH', hash_password('new-private-passphrase-123456'))
    monkeypatch.setenv('BLACKTHORN_ADMIN_PASSWORD', 'old-public-default')
    with TestClient(create_app(Settings(store='sqlite', sqlite_path=str(tmp_path / 'auth-only.sqlite')), gpu_daemon=False)) as client:
        origin = {'Origin': 'http://testserver'}
        assert client.post('/api/auth/login', json={'password': 'old-public-default'}, headers=origin).status_code == 401
        assert client.post('/api/auth/login', json={'password': 'new-private-passphrase-123456'}, headers=origin).status_code == 200
        assert client.post('/api/auth/reset', json={'current_password': 'old-public-default',
                    'new_password': 'another-private-passphrase-123456'}, headers=origin).status_code == 401


def test_production_auth_fails_closed_without_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv('BLACKTHORN_AUTH_REQUIRED', '1')
    monkeypatch.delenv('BLACKTHORN_ADMIN_PASSWORD', raising=False)
    monkeypatch.delenv('BLACKTHORN_BOOTSTRAP_PASSWORD', raising=False)
    monkeypatch.delenv('BLACKTHORN_AUTH_PASSWORD_HASH', raising=False)
    monkeypatch.setenv('BLACKTHORN_AUTH_SIGNING_KEY', 'z' * 48)
    with TestClient(create_app(Settings(store='sqlite', sqlite_path=str(tmp_path / 'empty.sqlite')), gpu_daemon=False)) as client:
        assert client.get('/api/auth/status').json()['configured'] is False
        assert client.get('/api/chat/sessions').status_code == 503
        assert client.post('/api/auth/login', json={'password': 'anything'},
                           headers={'Origin':'http://testserver'}).status_code == 503


def test_reset_hash_never_falls_back_to_old_password_on_storage_outage(monkeypatch, tmp_path):
    monkeypatch.setenv('BLACKTHORN_AUTH_REQUIRED', '1')
    monkeypatch.setenv('BLACKTHORN_AUTH_SIGNING_KEY', 'z' * 48)
    monkeypatch.setenv('BLACKTHORN_AUTH_PASSWORD_HASH', hash_password('obsolete-bootstrap-password'))
    with TestClient(create_app(Settings(store='sqlite', sqlite_path=str(tmp_path / 'outage.sqlite')), gpu_daemon=False)) as client:
        store = client.app.state.bt.store
        async def broken(_key):
            raise OSError('D1 unavailable')
        monkeypatch.setattr(store, 'get_setting', broken)
        assert client.post('/api/auth/login', json={'password': 'obsolete-bootstrap-password'},
                           headers={'Origin': 'http://testserver'}).status_code == 503
        assert client.get('/api/chat/sessions').status_code == 401
