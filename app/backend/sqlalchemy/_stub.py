from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Type

__all__ = [
    "JSON",
    "Float",
    "ForeignKey",
    "Integer",
    "String",
    "Text",
    "DefaultCallable",
    "func",
    "text",
    "select",
    "delete",
    "DeclarativeBase",
    "Mapped",
    "mapped_column",
    "relationship",
    "AsyncSession",
    "AsyncEngine",
    "async_sessionmaker",
    "create_async_engine",
]


class SQLType:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


JSON = SQLType("JSON")
Float = SQLType("Float")
Integer = SQLType("Integer")
String = SQLType("String")
Text = SQLType("Text")


class ForeignKey:
    def __init__(self, target: str, ondelete: Optional[str] = None) -> None:
        self.target = target
        self.ondelete = ondelete

    def __repr__(self) -> str:
        return f"ForeignKey({self.target!r})"


class DefaultCallable:
    def __init__(self, fn: Callable[[], Any]) -> None:
        self._fn = fn

    def __call__(self) -> Any:
        return self._fn()


class _Func:
    def now(self) -> DefaultCallable:
        return DefaultCallable(lambda: datetime.now(timezone.utc))


func = _Func()


class TextClause:
    def __init__(self, text: str) -> None:
        self.text = text

    def __str__(self) -> str:
        return self.text


def text(payload: str) -> TextClause:
    return TextClause(payload)


class Condition:
    def __init__(self, predicate: Callable[[Any], bool]) -> None:
        self.predicate = predicate

    def evaluate(self, item: Any) -> bool:
        try:
            return bool(self.predicate(item))
        except Exception:
            return False


class OrderClause:
    def __init__(self, column: "Column", descending: bool) -> None:
        self.column = column
        self.descending = descending

    def key(self, item: Any) -> Any:
        return getattr(item, self.column.name, None)


class Column:
    def __init__(self, *, default: Any = None, primary_key: bool = False) -> None:
        self.default = default
        self.primary_key = primary_key
        self.name: str = ""

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        self.name = name
        if not hasattr(owner, "__columns__"):
            owner.__columns__ = {}
        owner.__columns__[name] = self
        if self.primary_key:
            owner.__primary_key__ = name

    def default_value(self) -> Any:
        if isinstance(self.default, DefaultCallable):
            return self.default()
        if callable(self.default):
            return self.default()
        return self.default

    def __get__(self, instance: Any, owner: Type[Any]) -> Any:
        if instance is None:
            return self
        if self.name not in instance.__dict__:
            instance.__dict__[self.name] = self.default_value()
        return instance.__dict__[self.name]

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self.name] = value

    def __eq__(self, other: Any) -> Condition:  # type: ignore[override]
        return Condition(lambda item: getattr(item, self.name, None) == other)

    def __ne__(self, other: Any) -> Condition:  # type: ignore[override]
        return Condition(lambda item: getattr(item, self.name, None) != other)

    def asc(self) -> OrderClause:
        return OrderClause(self, descending=False)

    def desc(self) -> OrderClause:
        return OrderClause(self, descending=True)


class Relationship:
    def __init__(self) -> None:
        self.name: str = ""

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        self.name = name

    def __get__(self, instance: Any, owner: Type[Any]) -> Any:
        if instance is None:
            return self
        return instance.__dict__.setdefault(self.name, [])

    def __set__(self, instance: Any, value: Any) -> None:
        instance.__dict__[self.name] = list(value) if value is not None else []


class Select:
    def __init__(self, model: Type[Any]) -> None:
        self.model = model
        self.conditions: list[Condition] = []
        self.ordering: list[OrderClause] = []
        self._limit: Optional[int] = None

    def where(self, *conditions: Condition) -> "Select":
        for cond in conditions:
            if isinstance(cond, Condition):
                self.conditions.append(cond)
        return self

    def order_by(self, *clauses: OrderClause | Column) -> "Select":
        for clause in clauses:
            if isinstance(clause, OrderClause):
                self.ordering.append(clause)
            elif isinstance(clause, Column):
                self.ordering.append(OrderClause(clause, descending=False))
        return self

    def limit(self, value: Optional[int]) -> "Select":
        if value is None:
            self._limit = None
        else:
            self._limit = max(int(value), 0)
        return self


def select(model: Type[Any]) -> Select:
    return Select(model)


class Delete:
    def __init__(self, model: Type[Any]) -> None:
        self.model = model
        self.conditions: list[Condition] = []

    def where(self, *conditions: Condition) -> "Delete":
        for cond in conditions:
            if isinstance(cond, Condition):
                self.conditions.append(cond)
        return self


def delete(model: Type[Any]) -> Delete:
    return Delete(model)


class Mapped:
    def __class_getitem__(cls, item: Any) -> Any:
        return Any


class Metadata:
    def __init__(self) -> None:
        self._models: list[Type[Any]] = []

    def register(self, model: Type[Any]) -> None:
        if model not in self._models:
            self._models.append(model)

    def create_all(self, connection: Any) -> None:
        engine = getattr(connection, "engine", None)
        if engine is None:
            return
        for model in self._models:
            engine._ensure_model(model)


class DeclarativeMeta(type):
    def __new__(mcls, name: str, bases: Tuple[Type[Any], ...], namespace: Dict[str, Any]) -> Any:
        cls = super().__new__(mcls, name, bases, namespace)
        metadata = None
        for base in bases:
            metadata = getattr(base, "metadata", None)
            if metadata is not None:
                break
        if metadata is None:
            metadata = Metadata()
        cls.metadata = metadata
        if getattr(cls, "__tablename__", None):
            metadata.register(cls)
        if not hasattr(cls, "__columns__"):
            cls.__columns__ = {}
        if not hasattr(cls, "__primary_key__"):
            cls.__primary_key__ = None
        return cls


class DeclarativeBase(metaclass=DeclarativeMeta):
    metadata = Metadata()

    def __init__(self, **kwargs: Any) -> None:
        columns: Dict[str, Column] = getattr(self, "__columns__", {})
        for name, column in columns.items():
            if name in kwargs:
                value = kwargs[name]
            else:
                value = column.default_value()
            setattr(self, name, value)
        for key, value in kwargs.items():
            if key not in columns:
                setattr(self, key, value)


def mapped_column(*_: Any, default: Any = None, primary_key: bool = False, **__: Any) -> Column:
    return Column(default=default, primary_key=primary_key)


def relationship(*_: Any, **__: Any) -> Relationship:
    return Relationship()


class Result:
    def __init__(self, rows: Sequence[Any]) -> None:
        self._rows = list(rows)

    def scalar_one_or_none(self) -> Any:
        return self._rows[0] if self._rows else None

    def scalars(self) -> "ScalarResult":
        return ScalarResult(self._rows)

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class ScalarResult:
    def __init__(self, rows: Sequence[Any]) -> None:
        self._rows = list(rows)

    def all(self) -> list[Any]:
        return list(self._rows)

    def __iter__(self):
        return iter(self._rows)


class RawResult:
    def __init__(self, rows: Optional[Sequence[Any]] = None) -> None:
        self._rows = list(rows or [])

    def fetchall(self) -> list[Any]:
        return list(self._rows)


class _SyncConnection:
    def __init__(self, engine: "AsyncEngine") -> None:
        self.engine = engine


class AsyncConnection:
    def __init__(self, engine: "AsyncEngine") -> None:
        self.engine = engine
        self._sync = _SyncConnection(engine)

    async def __aenter__(self) -> "AsyncConnection":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def run_sync(self, fn: Callable[[Any], Any]) -> Any:
        if fn is None:
            return None
        return fn(self._sync)

    async def execute(self, statement: Any, params: Optional[dict] = None) -> RawResult:
        return RawResult([])


class AsyncEngine:
    def __init__(self, url: str | None = None, **_: Any) -> None:
        self.url = url
        self._store: Dict[Type[Any], list[Any]] = {}

    def begin(self) -> AsyncConnection:
        return AsyncConnection(self)

    def _ensure_model(self, model: Type[Any]) -> None:
        self._store.setdefault(model, [])


class AsyncSession:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine
        self._store = engine._store

    async def __aenter__(self) -> "AsyncSession":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def execute(self, statement: Any, params: Optional[dict] = None) -> Any:
        if isinstance(statement, Select):
            rows = list(self._store.get(statement.model, []))
            for cond in statement.conditions:
                rows = [row for row in rows if cond.evaluate(row)]
            for clause in reversed(statement.ordering):
                rows.sort(key=clause.key, reverse=clause.descending)
            if statement._limit is not None:
                rows = rows[: statement._limit]
            return Result(rows)
        if isinstance(statement, Delete):
            rows = list(self._store.get(statement.model, []))
            if statement.conditions:
                kept = [row for row in rows if not all(cond.evaluate(row) for cond in statement.conditions)]
            else:
                kept = []
            self._store[statement.model] = kept
            return RawResult([])
        if isinstance(statement, (str, TextClause)):
            return RawResult([])
        return RawResult([])

    async def get(self, model: Type[T], key: Any) -> Optional[T]:
        rows = self._store.get(model, [])
        pk = getattr(model, "__primary_key__", None)
        if pk is None:
            return None
        for row in rows:
            if getattr(row, pk, None) == key:
                return row
        return None

    def add(self, instance: Any) -> None:
        model = type(instance)
        self.engine._ensure_model(model)
        rows = self._store.setdefault(model, [])
        pk = getattr(model, "__primary_key__", None)
        if pk:
            key = getattr(instance, pk, None)
            for idx, row in enumerate(rows):
                if getattr(row, pk, None) == key:
                    rows[idx] = instance
                    break
            else:
                rows.append(instance)
        else:
            rows.append(instance)

    def add_all(self, instances: Iterable[Any]) -> None:
        for inst in instances:
            self.add(inst)

    async def commit(self) -> None:
        return None

    async def close(self) -> None:
        return None


class SessionFactory:
    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    def __call__(self) -> AsyncSession:
        return AsyncSession(self.engine)


def create_async_engine(url: str, **kwargs: Any) -> AsyncEngine:
    return AsyncEngine(url, **kwargs)


def async_sessionmaker(*, bind: AsyncEngine, **kwargs: Any) -> SessionFactory:
    return SessionFactory(bind)
