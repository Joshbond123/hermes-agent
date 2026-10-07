"""Blackthorn — the product layer on top of the Hermes dashboard.

Everything that is specific to the Blackthorn deployment (Cloudflare D1 storage, the
Kaggle GPU controller, the streaming agent engine and its HTTP API) lives in this
package and is committed to GitHub like the rest of the application.  GitHub is the
single source of truth: nothing in here is ever downloaded or patched in at build or
boot time.
"""

__all__ = ["config", "d1"]
