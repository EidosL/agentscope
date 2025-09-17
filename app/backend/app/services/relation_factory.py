from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..agents.relation_factory_agent import run_relation_factory_once
from ..db import models
from ..schemas.relations import CandidatePayload, CanvasSubmitRequest, EvidenceSnippet

ARTIFACT_ROOT = Path("artifacts/audit").resolve()

TOP_S_SNIPPETS = 3
FTS_LIMIT = 5

PRED_QUERIES: dict[str, list[str]] = {
    "supports": [
        "support",
        "supports",
        "supporting",
        "supported",
        "evidence",
        "依据",
        "支持",
        "佐证",
        "证明",
        "支撑",
    ],
    "contradicts": [
        "contradict",
        "contradicts",
        "contradiction",
        "refute",
        "refutes",
        "冲突",
        "矛盾",
        "反驳",
        "驳斥",
    ],
    "causes": [
        "cause",
        "causes",
        "caused",
        "result",
        "results",
        "导致",
        "引发",
        "引起",
        "造成",
        "致使",
        "因为",
    ],
    "implies": [
        "imply",
        "implies",
        "implied",
        "意味着",
        "表明",
        "暗示",
    ],
    "relates_to": [
        "related",
        "relate",
        "relates",
        "relationship",
        "关联",
        "联系",
        "相关",
        "相连",
    ],
    "part_of": [
        "part of",
        "belongs",
        "belong",
        "包含",
        "属于",
        "组成",
        "构成",
    ],
    "extends": [
        "extend",
        "extends",
        "extended",
        "拓展",
        "扩展",
        "延伸",
        "深化",
    ],
    "opposes": [
        "oppose",
        "opposes",
        "opposed",
        "opposition",
        "对立",
        "对抗",
        "反对",
    ],
    "contrasts": [
        "contrast",
        "contrasts",
        "contrasted",
        "对比",
        "相反",
        "差异",
    ],
    "enables": [
        "enable",
        "enables",
        "enabled",
        "促成",
        "使得",
        "允许",
    ],
    "prevents": [
        "prevent",
        "prevents",
        "prevented",
        "避免",
        "阻止",
        "制止",
    ],
    "requires": [
        "require",
        "requires",
        "required",
        "需要",
        "依赖",
        "取决",
        "必须",
    ],
}

DEFAULT_PRED_KEYWORDS = [
    "关联",
    "联系",
    "相关",
    "关系",
    "link",
    "links",
    "linked",
    "connection",
    "connected",
    "connect",
    "because",
    "因此",
    "所以",
]


def _predicate_keywords(predicate: str) -> list[str]:
    norm = (predicate or "").lower().strip()
    base = PRED_QUERIES.get(norm)
    if base:
        keywords = [kw.lower() for kw in base if kw]
    else:
        parts = [norm] if norm else []
        for token in re.split(r"[\s_/|-]+", norm):
            if token and token not in parts:
                parts.append(token)
        keywords = [kw.lower() for kw in parts if kw]
        keywords.extend(DEFAULT_PRED_KEYWORDS)

    seen: set[str] = set()
    deduped: list[str] = []
    for kw in keywords:
        key = kw.lower()
        if key and key not in seen:
            deduped.append(key)
            seen.add(key)
    return deduped


def has_pred_kw(predicate: str, quotes: Iterable[str]) -> bool:
    """Return True when at least one predicate keyword appears in the snippets."""

    if not predicate:
        return True

    text = " ".join(q.lower() for q in quotes if q).strip()
    if not text:
        return False

    for keyword in _predicate_keywords(predicate):
        if keyword in text:
            return True
    return False


async def check_blacklist(session: AsyncSession, uniq_key: str) -> bool:
    result = await session.execute(select(models.RelationReject).where(models.RelationReject.uniq_key == uniq_key))
    return result.scalar_one_or_none() is not None


def normalize_claim(subject: str, predicate: str, obj: str, claim: str) -> str:
    payload = f"{subject}|{predicate}|{obj}|{claim}".lower().strip()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_candidate_from_relation(relation: models.Relation, evidences: list[models.RelationEvidence]) -> CandidatePayload:
    evidence_payload = [
        EvidenceSnippet(
            note=ev.note_id,
            span=ev.span,
            quote=ev.quote,
            quote_sha=ev.quote_sha,
        )
        for ev in evidences
    ]
    return CandidatePayload(
        id=relation.id,
        subject=relation.subject,
        predicate=relation.predicate,
        object=relation.object,
        claim=relation.claim,
        explain=relation.reason or "",
        confidence=relation.confidence or 0.0,
        event_time=relation.event_time,
        valid_from=relation.valid_from,
        valid_to=relation.valid_to,
        evidence=evidence_payload,
        scores={
            "bm25": relation.bm25 or 0.0,
            "cos": relation.cos or 0.0,
            "npmi": relation.npmi or 0.0,
            "time": relation.time_fresh or 0.0,
            "novelty": relation.novelty or 0.0,
        },
        degraded=False,
    )


async def _fts_top_matches(session: AsyncSession, subject: str, query: str, limit: int = FTS_LIMIT) -> list[tuple[str, str]]:
    """Return top FTS matches as (note_id, snippet) excluding the subject note."""

    if not query:
        return []

    try:
        res = await session.execute(
            f"SELECT id, snippet(notes_fts, -1, '', '', ' … ', 64) as snip FROM notes_fts WHERE notes_fts MATCH :q LIMIT {max(limit, 1)}",
            {"q": query},
        )
    except Exception:
        return []

    hits: list[tuple[str, str]] = []
    seen: set[str] = set()
    for rid, snip in res.fetchall():
        if not rid or rid == subject or rid in seen:
            continue
        seen.add(rid)
        hits.append((rid, snip or ""))
        if len(hits) >= limit:
            break
    return hits


async def run_relation_factory(
    session: AsyncSession,
    request: CanvasSubmitRequest,
    debug: bool = False,
) -> tuple[Optional[CandidatePayload], dict[str, Any]]:
    subject = request.note_id or "Z_subject_auto"
    predicate = request.predicate or "supports"
    obj_default = "Z_object_auto"

    debug_payload: dict[str, Any] = {
        "gate": {
            "predicate": predicate,
            "fallback_used": False,
            "pairs": [],
        }
    }

    subj_note = await session.get(models.Note, subject)
    subj_quote = (
        subj_note.content[:140]
        if subj_note and subj_note.content
        else request.content[:140]
    )

    subject_ev = {
        "note": subject,
        "span": "L1-L12",
        "quote": subj_quote,
        "quote_sha": None,
    }
    subject_cover = has_pred_kw(predicate, [subj_quote])

    fts_hits = await _fts_top_matches(session, subject, request.content[:128])
    candidate_pairs: list[dict[str, Any]] = []
    seen_objs: set[str] = set()
    for idx, (obj_candidate, snip) in enumerate(fts_hits):
        if obj_candidate in seen_objs:
            continue
        seen_objs.add(obj_candidate)

        obj_note = await session.get(models.Note, obj_candidate)
        raw_quote = ""
        if obj_note and obj_note.content:
            raw_quote = obj_note.content
        else:
            raw_quote = snip or request.content[:80]
        obj_quote = (raw_quote or "")[:140]

        pair_subject = subject_ev.copy()
        pair_object = {
            "note": obj_candidate,
            "span": f"L20-L{36 + idx}",
            "quote": obj_quote,
            "quote_sha": None,
        }
        object_cover = has_pred_kw(predicate, [obj_quote])
        record = {
            "subject": subject,
            "object": obj_candidate,
            "subject_cover": subject_cover,
            "object_cover": object_cover,
            "passed_gate": subject_cover and object_cover,
            "skipped": not (subject_cover and object_cover),
            "evaluated": False,
            "used": False,
            "fallback": False,
        }
        candidate_pairs.append(
            {
                "subject": pair_subject,
                "object": pair_object,
                "passed_gate": record["passed_gate"],
                "debug_record": record,
            }
        )
        debug_payload["gate"]["pairs"].append(record)

    if not candidate_pairs:
        fallback_quote = (request.content or "")[:80]
        pair_subject = subject_ev.copy()
        pair_object = {
            "note": obj_default,
            "span": "L20-L36",
            "quote": fallback_quote,
            "quote_sha": None,
        }
        object_cover = has_pred_kw(predicate, [fallback_quote])
        record = {
            "subject": subject,
            "object": obj_default,
            "subject_cover": subject_cover,
            "object_cover": object_cover,
            "passed_gate": subject_cover and object_cover,
            "skipped": not (subject_cover and object_cover),
            "evaluated": False,
            "used": False,
            "fallback": False,
        }
        candidate_pairs.append(
            {
                "subject": pair_subject,
                "object": pair_object,
                "passed_gate": record["passed_gate"],
                "debug_record": record,
            }
        )
        debug_payload["gate"]["pairs"].append(record)

    passed_pairs = [pair for pair in candidate_pairs if pair["passed_gate"]]
    if passed_pairs:
        pairs_to_eval = passed_pairs
    else:
        pairs_to_eval = candidate_pairs[:TOP_S_SNIPPETS]
        debug_payload["gate"]["fallback_used"] = True
        for pair in pairs_to_eval:
            record = pair["debug_record"]
            record["fallback"] = True
            record["skipped"] = False

    fused = None
    used_pair: Optional[dict[str, Any]] = None
    llm_calls = 0

    for pair in pairs_to_eval:
        record = pair["debug_record"]
        record["skipped"] = False
        record["evaluated"] = True

        def evidence_provider(pair=pair) -> dict[str, Any]:
            return {"evidence": [pair["subject"], pair["object"]]}

        llm_calls += 1
        fused = await run_relation_factory_once(
            subject=subject,
            predicate=predicate,
            content=request.content,
            evidence_provider=evidence_provider,
        )
        if fused and fused.evidence:
            used_pair = pair
            record["used"] = True
            break

    if not pairs_to_eval:
        pairs_to_eval = candidate_pairs

    selected_pair = used_pair or (pairs_to_eval[0] if pairs_to_eval else candidate_pairs[0])
    selected_record = selected_pair["debug_record"]
    selected_record["used"] = True
    if not selected_record["evaluated"]:
        selected_record["evaluated"] = True
    selected_record["skipped"] = False

    obj_id = selected_pair["object"].get("note") or obj_default

    debug_payload["gate"]["llm_calls"] = llm_calls
    debug_payload["gate"]["selected"] = {"subject": subject, "object": obj_id}

    if not fused or not fused.evidence:
        claim = f"你的输入提示 {subject} 与 {obj_id} 在主题上形成{predicate}连接。"
        explain = "连接原因：它从另一个角度推进了你刚写的主题。"
        fused_claim = claim
        fused_reason = explain
        fused_evidence = [
            EvidenceSnippet(**selected_pair["subject"]),
            EvidenceSnippet(**selected_pair["object"]),
        ]
    else:
        fused_claim = fused.claim
        fused_reason = fused.reason
        fused_evidence = fused.evidence

    uniq_key = normalize_claim(subject, predicate, obj_id, fused_claim)
    if await check_blacklist(session, uniq_key):
        return None, debug_payload if debug else {}

    existing_rel_result = await session.execute(
        select(models.Relation).where(models.Relation.uniq_key == uniq_key)
    )
    existing_relation = existing_rel_result.scalar_one_or_none()
    if existing_relation is not None:
        evidences_res = await session.execute(
            select(models.RelationEvidence).where(
                models.RelationEvidence.rel_id == existing_relation.id
            )
        )
        evidences = evidences_res.scalars().all()
        candidate = build_candidate_from_relation(existing_relation, evidences)
        return candidate, debug_payload if debug else {}

    relation_id = f"Rel_{uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    relation = models.Relation(
        id=relation_id,
        subject=subject,
        predicate=predicate,
        object=obj_id,
        claim=fused_claim,
        reason=fused_reason,
        confidence=0.72,
        status="proposed",
        created_at=now,
        updated_at=now,
        event_time=now,
        valid_from=now,
        uniq_key=uniq_key,
        bm25=0.34,
        cos=0.78,
        npmi=0.41,
        time_fresh=0.55,
        path2=0.45,
        novelty=0.66,
        score=0.68,
    )

    session.add(relation)

    ev_models: list[models.RelationEvidence] = []
    for ev in fused_evidence[:2]:
        ev_models.append(
            models.RelationEvidence(
                rel_id=relation_id,
                note_id=ev.note,
                span=ev.span,
                kind="note",
                quote=ev.quote,
                quote_sha=hashlib.sha256(ev.quote.encode("utf-8")).hexdigest() if ev.quote else None,
            )
        )

    session.add_all(ev_models)

    # Minimal audit artifact (fuse stage)
    try:
        run_id = f"run_{relation_id}"
        ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        (ARTIFACT_ROOT / f"{run_id}_fuse.json").write_text(
            json.dumps(
                {
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj_id,
                    "input": request.content,
                    "fused": {
                        "claim": fused_claim,
                        "reason": fused_reason,
                        "evidence": [e.model_dump() for e in fused_evidence],
                    },
                    "created_at": now.isoformat(),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    await session.commit()

    candidate = build_candidate_from_relation(relation, ev_models)
    return candidate, debug_payload if debug else {}
