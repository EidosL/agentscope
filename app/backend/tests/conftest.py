import asyncio
import inspect

import httpx
import pytest


if "app" not in inspect.signature(httpx.AsyncClient.__init__).parameters:
    _orig_async_client_init = httpx.AsyncClient.__init__

    def _patched_async_client_init(self, *args, app=None, **kwargs):
        if app is not None:
            transport = kwargs.get("transport")
            if transport is None:
                kwargs["transport"] = httpx.ASGITransport(app=app)
        return _orig_async_client_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _patched_async_client_init


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "asyncio: mark test functions as asynchronous to run via a simple event loop runner.",
    )


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem: pytest.Function) -> bool:
    if asyncio.iscoroutinefunction(pyfuncitem.obj):
        kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in pyfuncitem._fixtureinfo.argnames
            if name in pyfuncitem.funcargs
        }
        asyncio.run(pyfuncitem.obj(**kwargs))
        return True
    return False
