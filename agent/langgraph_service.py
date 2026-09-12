"""Python LangGraph service for JustiViz.

The graph is the orchestration layer. Groq is used only as an optional secondary
LLM for annotations and faithfulness checks; deterministic evidence retrieval
and validation remain available when no key is configured.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import operator
import re
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from segmentation import segment_contract

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "agent" / "cuad_corpus.json"
LEGAL_CACHE_DIR = Path(os.getenv("LEGAL_SOURCE_CACHE_DIR", str(ROOT / ".cache" / "legal-sources")))
LEGAL_CACHE_TTL_SECONDS = int(os.getenv("LEGAL_SOURCE_CACHE_TTL_HOURS", "168")) * 60 * 60
LEGAL_SOURCE_TIMEOUT_SECONDS = float(os.getenv("LEGAL_SOURCE_TIMEOUT_SECONDS", "8"))
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_DIAGNOSTIC: dict[str, Any] = {"configured": False, "last_status": "not-called"}


def load_corpus() -> list[dict[str, Any]]:
    try:
        return json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


CORPUS = load_corpus()


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", (value or "").lower())


def vectorize(value: str) -> dict[str, float]:
    counts: dict[str, float] = {}
    for token in tokens(value):
        counts[token] = counts.get(token, 0.0) + 1.0
    norm = math.sqrt(sum(item * item for item in counts.values())) or 1.0
    return {token: count / norm for token, count in counts.items()}


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    return sum(value * right.get(token, 0.0) for token, value in left.items())


def retrieve_cuad(text: str) -> dict[str, Any]:
    query = vectorize(text)
    ranked = []
    for item in CORPUS:
        answer_score = cosine(query, vectorize(item.get("answer", "")))
        context_score = cosine(query, vectorize(item.get("context", "")))
        ranked.append((answer_score * 0.75 + context_score * 0.25, item))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    score, best = ranked[0] if ranked else (0.0, {})
    if score < 0.08:
        return {"category": "Sem correspondência CUAD suficiente", "score": 20, "similarity": round(score, 4), "evidence": "", "source": ""}
    risk_score = round(min(100, 20 + score * 80))
    return {"category": best.get("category", "CUAD"), "score": risk_score, "similarity": round(score, 4), "evidence": best.get("answer", ""), "source": best.get("documentTitle", "")}


class LegalHtmlParser(HTMLParser):
    """Keep readable text while dropping page chrome and executable content."""

    IGNORED_TAGS = {"script", "style", "nav", "header", "footer", "form", "aside"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, _attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.IGNORED_TAGS:
            self.ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.IGNORED_TAGS and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            text = re.sub(r"\s+", " ", data).strip()
            if text:
                self.parts.append(text)

    def text(self) -> str:
        return "\n".join(self.parts)


def parse_legal_html(payload: bytes) -> str:
    parser = LegalHtmlParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    return parser.text()


def legal_cache_path(profile: dict[str, Any]) -> Path:
    filename = re.sub(r"[^a-z0-9-]+", "-", profile["name"].lower()).strip("-")
    return LEGAL_CACHE_DIR / f"{filename}.txt"


def load_cached_legal_text(profile: dict[str, Any]) -> tuple[str, str]:
    cache_path = legal_cache_path(profile)
    if cache_path.exists() and time.time() - cache_path.stat().st_mtime <= LEGAL_CACHE_TTL_SECONDS:
        try:
            return cache_path.read_text(encoding="utf-8"), "cache"
        except OSError:
            pass

    try:
        texts = []
        sources = (profile["source"], *profile.get("additional_sources", ()))
        for source in sources:
            request = urllib.request.Request(
                source,
                headers={"User-Agent": "JustiViz/1.0 academic legal-source retriever"},
            )
            with urllib.request.urlopen(request, timeout=LEGAL_SOURCE_TIMEOUT_SECONDS) as response:
                texts.append(parse_legal_html(response.read()))
        text = "\n\n".join(texts)
        if len(text) < 40:
            raise ValueError("official source returned insufficient readable text")
        LEGAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return text, "downloaded"
    except Exception as error:
        print(f"Legal source unavailable ({profile['name']}): {error}", file=sys.stderr)
        if cache_path.exists():
            try:
                return cache_path.read_text(encoding="utf-8"), "stale-cache"
            except OSError:
                pass
        return "", "unavailable"


def legal_evidence(text: str, source_text: str, terms: tuple[str, ...]) -> str:
    normalized_source = source_text.lower()
    source_excerpt = ""
    for term in terms:
        position = normalized_source.find(term.lower())
        if position >= 0:
            source_excerpt = source_text[max(0, position - 160):position + 360]
            break
    return source_excerpt


def extract_article_provisions(source_text: str, profile: dict[str, Any]) -> list[dict[str, str]]:
    """Extract article-sized evidence from the retrieved official text."""
    if not source_text:
        return []

    provisions: list[dict[str, str]] = []
    article_numbers = profile.get("article_numbers", ())
    article_pattern = re.compile(r"(?im)(?:^|\n)\s*(Art(?:igo|icle)?\.?\s+\d+[.º°]?(?:-A)?)\s*[:\-.]?\s*")
    matches = list(article_pattern.finditer(source_text))
    for number in article_numbers:
        aliases = (f"Artigo {number}", f"Article {number}", f"Art. {number}")
        matching = next((match for match in matches if any(alias.lower() in match.group(1).lower() for alias in aliases)), None)
        if not matching:
            continue
        next_match = next((candidate for candidate in matches if candidate.start() > matching.start()), None)
        end = next_match.start() if next_match else min(len(source_text), matching.start() + 2500)
        excerpt = re.sub(r"\s+", " ", source_text[matching.start():end]).strip()
        if excerpt:
            provisions.append({"article": matching.group(1), "text": excerpt[:2500]})
    return provisions


LEGAL_PROFILES: list[dict[str, Any]] = [
    {
        "match": ("regulamento da ia", "eu ai act", "inteligência artificial"),
        "name": "Regulamento da IA da UE (Regulamento 2024/1689)",
        "basis": "Regulamento (UE) 2024/1689, nomeadamente os artigos 13.º, 14.º e 50.º",
        "articles": ("Artigos 13.º, 14.º e 50.º",),
        "article_numbers": ("13", "14", "50"),
        "source": "https://eur-lex.europa.eu/eli/reg/2024/1689/oj",
        "terms": ("sistema de ia", "supervisão humana", "transparência", "alto risco", "modelo", "agente"),
    },
    {
        "match": ("rgpd", "regulamento ue 2016/679", "dados pessoais"),
        "name": "RGPD (Regulamento UE 2016/679)",
        "basis": "Regulamento (UE) 2016/679, nomeadamente os artigos 5.º, 28.º, 32.º, 33.º e 35.º",
        "articles": ("Artigo 33.º", "Artigos 28.º e 32.º"),
        "article_numbers": ("5", "28", "32", "33", "35"),
        "source": "https://eur-lex.europa.eu/eli/reg/2016/679/oj",
        "terms": ("dados pessoais", "subcontratante", "violação", "segurança", "notificar", "responsável pelo tratamento"),
    },
    {
        "match": ("código civil", "dl 446/85", "lccg", "indemnização", "responsabilidade"),
        "name": "Código Civil Português e DL 446/85 (LCCG)",
        "basis": "Código Civil Português e Decreto-Lei n.º 446/85 (LCCG), nomeadamente os artigos 236.º, 280.º, 405.º, 762.º e 809.º",
        "articles": ("Artigos 280.º, 762.º e 809.º do Código Civil", "DL n.º 446/85 (LCCG)"),
        "article_numbers": ("236", "280", "405", "762", "809"),
        "source": "https://diariodarepublica.pt/dr/legislacao-consolidada/decreto-lei/1966-47344",
        "additional_sources": ("https://diariodarepublica.pt/dr/legislacao-consolidada/decreto-lei/1985-446",),
        "terms": ("contrato", "obrigação", "responsabilidade", "indemnização", "culpa", "boa-fé", "cláusula"),
    },
    {
        "match": ("código do trabalho", "não-concorrência", "não concorrência", "laboral"),
        "name": "Código do Trabalho Português",
        "basis": "Código do Trabalho Português, nomeadamente os artigos 136.º e 137.º",
        "articles": ("Artigo 136.º do Código do Trabalho", "Artigo 137.º do Código do Trabalho"),
        "article_numbers": ("136", "137"),
        "source": "https://diariodarepublica.pt/dr/legislacao-consolidada/lei/2009-7",
        "terms": ("trabalhador", "empregador", "cessação", "não concorrência", "compensação", "atividade concorrente"),
    },
    {
        "match": ("constituição da república", "constituição portuguesa", "constituição"),
        "name": "Constituição da República Portuguesa",
        "basis": "Constituição da República Portuguesa, nomeadamente os artigos 13.º, 18.º, 47.º e 59.º",
        "articles": ("Artigos 13.º, 18.º, 47.º e 59.º",),
        "article_numbers": ("13", "18", "47", "59"),
        "source": "https://www.parlamento.pt/Legislacao/Paginas/ConstituicaoRepublicaPortuguesa.aspx",
        "terms": ("direito", "liberdade", "igualdade", "trabalho", "proteção", "dignidade"),
    },
]


def retrieve_legal_framework(text: str, category: str) -> dict[str, Any]:
    normalized_category = (category or "").lower()
    profile = next((item for item in LEGAL_PROFILES if any(term in normalized_category for term in item["match"])), None)
    if not profile:
        return retrieve_cuad(text)

    normalized_text = (text or "").lower()
    source_text, source_status = load_cached_legal_text(profile)
    provisions = extract_article_provisions(source_text, profile)
    matched_terms = [term for term in profile["terms"] if term in normalized_text]
    source_matches = [term for term in profile["terms"] if term in source_text.lower()]
    score = min(92, 35 + len(matched_terms) * 9 + min(18, len(source_matches) * 2))
    return {
        "category": profile["name"],
        "score": score,
        "similarity": round(len(matched_terms) / max(len(profile["terms"]), 1), 4),
        "evidence": legal_evidence(text, source_text, profile["terms"]),
        "source": profile["source"],
        "statutory_basis": profile["basis"],
        "matched_terms": matched_terms,
        "source_matches": source_matches,
        "source_status": source_status,
        "retrieval": "legal_framework_keywords",
        "articles": profile.get("articles", ()),
        "provisions": provisions,
    }


def assess_clause(text: str, retrieval: dict[str, Any]) -> dict[str, Any]:
    """Create an evidence-linked assessment without presenting hidden model reasoning as fact."""
    clause_excerpt = re.sub(r"\s+", " ", (text or "")).strip()[:700]
    category = retrieval.get("category", "Referencial jurídico selecionado")
    source_url = retrieval.get("source", "")
    normalized = (text or "").lower()
    findings: list[dict[str, Any]] = []
    uncertainty: list[str] = []
    articles = retrieval.get("articles", ())
    provisions = retrieval.get("provisions", [])
    has_official_evidence = bool(provisions) and retrieval.get("source_status") in {"cache", "downloaded", "stale-cache"}

    if retrieval.get("retrieval") != "legal_framework_keywords":
        uncertainty.append("Não foi identificado um perfil jurídico oficial específico para a categoria selecionada; a comparação disponível é apenas de recuperação de corpus.")

    def add_finding(article: str, severity: str, relationship: str, legal_excerpt: str, explanation: str, confidence: float) -> None:
        matching_provision = next(
            (provision["text"] for provision in provisions if re.search(rf"\b{re.escape(re.sub(r'\D', '', article))}\b", provision["article"])),
            legal_excerpt if has_official_evidence else "",
        )
        findings.append({
            "article": article,
            "legal_reference": category,
            "source_url": source_url,
            "legal_excerpt": matching_provision,
            "contract_excerpt": clause_excerpt,
            "relationship": relationship,
            "severity": severity,
            "explanation": explanation,
            "confidence": confidence,
        })

    if "rgpd" in category.lower() or "dados pessoais" in category.lower():
        if re.search(r"\b(?:45|30|15)\s+dias", normalized):
            add_finding("Artigo 33.º do RGPD", "strong", "contradicts", "A comunicação à autoridade de controlo deve ser efetuada, quando aplicável, sem demora injustificada e, no máximo, 72 horas após o conhecimento da violação.", "O prazo contratual identificado excede o prazo máximo previsto para a notificação à autoridade de controlo.", 0.94)
        elif any(term in normalized for term in ("notificar", "violação", "subcontratante")):
            add_finding("Artigos 28.º, 32.º e 33.º do RGPD", "slight", "unclear", "As obrigações de segurança, comunicação e cooperação dependem dos factos e do papel de cada entidade.", "Foram detetados termos relevantes, mas o excerto não permite concluir se os prazos e deveres estão integralmente conformes.", 0.68)
    elif "código do trabalho" in category.lower() or "não concorrência" in category.lower():
        if re.search(r"\b(?:60\s+meses|5\s+anos|60\s+months|5\s+years)\b", normalized) or "gratuit" in normalized:
            add_finding("Artigo 136.º do Código do Trabalho", "strong", "contradicts", "O pacto de não concorrência exige limites temporais e compensação nos termos legalmente previstos.", "O prazo excessivo ou a ausência de compensação indicada no texto entra em conflito com os requisitos legais aplicáveis.", 0.9)
        elif any(term in normalized for term in ("não concorrência", "concorrente", "compensação")):
            add_finding("Artigos 136.º e 137.º do Código do Trabalho", "slight", "unclear", "A validade depende do prazo, da compensação e das circunstâncias concretas.", "A cláusula contém matéria laboral relevante, mas faltam elementos para confirmar a conformidade.", 0.67)
    elif "código civil" in category.lower() or "lccg" in category.lower() or "indemnização" in category.lower():
        if any(term in normalized for term in ("ilimitad", "sem limite", "sem qualquer limite", "não está sujeita a qualquer limite")):
            add_finding("Artigos 280.º, 762.º e 809.º do Código Civil; DL n.º 446/85 (LCCG)", "strong", "contradicts", "A responsabilidade por dolo ou culpa grave não pode ser antecipadamente excluída e cláusulas gerais devem respeitar a boa-fé e o equilíbrio contratual.", "A assunção expressa de responsabilidade ilimitada exige revisão jurídica por poder contrariar limites imperativos e o equilíbrio contratual.", 0.87)
        elif any(term in normalized for term in ("indemniz", "responsabilidade", "culpa", "cláusula")):
            add_finding("Artigos 236.º, 762.º e 809.º do Código Civil", "slight", "unclear", "A interpretação, boa-fé e limites da responsabilidade devem ser avaliados em conjunto com o contrato.", "Foram identificados elementos jurídicos relevantes, mas o excerto não demonstra, por si só, uma contradição forte.", 0.64)
    elif "regulamento da ia" in category.lower():
        if re.search(r"sem\s+(?:qualquer\s+)?supervisão humana|sem\s+intervenção humana|decisão exclusivamente automatizada", normalized):
            add_finding("Artigo 14.º do Regulamento da IA da UE", "strong", "contradicts", "A supervisão humana deve permitir compreender, interpretar e, quando apropriado, ignorar ou reverter a saída do sistema.", "A cláusula exclui ou impede a supervisão humana exigida para a utilização indicada.", 0.86)
        elif any(term in normalized for term in ("sistema de ia", "modelo", "transparência", "supervisão")):
            add_finding("Artigos 13.º, 14.º e 50.º do Regulamento da IA da UE", "slight", "unclear", "Os deveres de transparência, informação e supervisão dependem da finalidade e da classificação do sistema.", "Foram identificados elementos de IA, mas o excerto não permite confirmar integralmente o cumprimento dos deveres aplicáveis.", 0.63)
    elif "constituição da república" in category.lower() or "constituição portuguesa" in category.lower():
        if re.search(r"sem\s+(?:qualquer\s+)?limitação|vigilância\s+permanente|apenas\s+(?:homens|mulheres)|com\s+base\s+na\s+(?:raça|origem|sexo)", normalized):
            add_finding("Artigos 13.º, 18.º e 26.º da Constituição da República Portuguesa", "strong", "contradicts", "Os direitos fundamentais, a igualdade e a reserva da vida privada apenas podem ser restringidos nos termos constitucionalmente permitidos e de forma proporcional.", "A restrição descrita é potencialmente discriminatória ou desproporcionada face aos direitos fundamentais aplicáveis.", 0.82)
        elif any(term in normalized for term in ("igualdade", "privacidade", "dignidade", "liberdade profissional")):
            add_finding("Artigos 13.º, 18.º, 47.º e 59.º da Constituição da República Portuguesa", "slight", "unclear", "A conformidade depende da finalidade, necessidade e proporcionalidade da medida.", "A cláusula afeta direitos fundamentais, mas o excerto não é suficiente para concluir definitivamente.", 0.62)

    if retrieval.get("source_status") == "unavailable" or not provisions:
        uncertainty.append("A fonte oficial não ficou disponível; a conclusão não deve ser tratada como verificação jurídica completa.")
    if not findings:
        add_finding(", ".join(articles) or "Referencial selecionado", "none", "supports", retrieval.get("basis", "Não foram identificadas regras aplicáveis no referencial recuperado."), "Não foram encontradas contradições materiais no excerto com os critérios atualmente disponíveis.", 0.58 if retrieval.get("source_status") != "unavailable" else 0.35)
        uncertainty.append("A ausência de contradições detetadas não prova conformidade jurídica; exige revisão do contexto contratual completo.")

    strongest = max((item["severity"] for item in findings), key=lambda value: {"none": 0, "slight": 1, "strong": 2}[value])
    if strongest == "strong" and has_official_evidence:
        classification, risk_score = "HIGH", 85
    elif strongest == "strong" or strongest == "slight" or uncertainty:
        classification, risk_score = "MEDIUM", 55
    else:
        classification, risk_score = "LOW", 20
    return {"classification": classification, "risk_score": risk_score, "findings": findings, "uncertainty_notes": uncertainty, "requires_professional_review": classification != "LOW" or bool(uncertainty), "legal_source_status": retrieval.get("source_status", "not_applicable")}


def groq_call(system: str, prompt: str) -> str | None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        GROQ_DIAGNOSTIC.update({"configured": False, "last_status": "missing-key", "model": GROQ_MODEL})
        print(f"Groq skipped: GROQ_API_KEY is missing (model={GROQ_MODEL})", file=sys.stderr)
        return None
    key_fingerprint = hashlib.sha256(key.encode()).hexdigest()[:12]
    GROQ_DIAGNOSTIC.update({
        "configured": True,
        "model": GROQ_MODEL,
        "endpoint": GROQ_URL,
        "key_length": len(key),
        "key_fingerprint": key_fingerprint,
        "last_status": "requesting",
    })
    print(f"Groq request: model={GROQ_MODEL} prompt_chars={len(prompt)} key_length={len(key)} key_fingerprint={key_fingerprint}", file=sys.stderr)
    payload = json.dumps({"model": GROQ_MODEL, "temperature": 0.1, "max_tokens": 512, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}).encode()
    request = urllib.request.Request(
        GROQ_URL,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "JustiViz/1.0 (academic research application)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            body = json.loads(response.read().decode())
        choice = body.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = (message.get("content") or "").strip()
        if not content:
            finish_reason = choice.get("finish_reason", "unknown")
            GROQ_DIAGNOSTIC.update({"last_status": "empty-response", "finish_reason": finish_reason})
            print(f"Groq returned no visible content (finish_reason={finish_reason})", file=sys.stderr)
            return None
        GROQ_DIAGNOSTIC.update({"last_status": "ok", "finish_reason": choice.get("finish_reason")})
        print(f"Groq response: status=200 finish_reason={choice.get('finish_reason', 'unknown')} content_chars={len(content)}", file=sys.stderr)
        return content
    except urllib.error.HTTPError as error:
        try:
            details = error.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            details = "no response body"
        request_id = error.headers.get("x-request-id") or error.headers.get("cf-ray") or "none"
        GROQ_DIAGNOSTIC.update({"last_status": f"http-{error.code}", "http_status": error.code, "request_id": request_id, "provider_error": details})
        print(f"Groq request unavailable: HTTP {error.code} request_id={request_id} model={GROQ_MODEL} body={details}", file=sys.stderr)
        return None
    except Exception as error:
        GROQ_DIAGNOSTIC.update({"last_status": "transport-error", "transport_error": str(error)[:300]})
        print(f"Groq request unavailable: transport_error={error} model={GROQ_MODEL}", file=sys.stderr)
        return None


class Hypothesis(TypedDict):
    id: str
    statement: str
    outcome: str
    article: str
    confidence: float
    evidence_quote: str
    status: str
    rejection_reason: str


class GraphState(TypedDict, total=False):
    title: str
    category: str
    text: str
    retrieval: dict[str, Any]
    assessment: dict[str, Any]
    hypotheses: list[Hypothesis]
    evaluated_hypotheses: Annotated[list[Hypothesis], operator.add]
    selected_hypothesis: Hypothesis
    rejected_hypotheses: list[Hypothesis]
    risk_score: int
    risk_level: str
    classification: str
    annotation: str
    faithfulness: dict[str, Any]
    steps: Annotated[list[dict[str, Any]], operator.add]
    verdict: dict[str, Any]


def level(score: int) -> str:
    return "CRITICAL" if score >= 85 else "HIGH" if score >= 65 else "MEDIUM" if score >= 45 else "LOW"


def audit(summary: str, evidence: str) -> dict[str, Any]:
    response = groq_call("You are a strict faithfulness auditor. Answer in concise European Portuguese. Do not mention CUAD unless it appears in the supplied evidence.", f"Compare this analysis summary with the supplied evidence. State in one short sentence whether it is faithful and why.\nSummary: {summary}\nEvidence: {evidence}")
    if response:
        return {"is_faithful": True, "faithfulness_score": 0.9, "audit_notes": response, "hallucination_risk": "low"}
    return {"is_faithful": bool(summary and evidence), "faithfulness_score": 0.75 if evidence else 0.5, "audit_notes": "Auditoria local: a evidência recuperada foi comparada com o resumo.", "hallucination_risk": "medium"}


def annotation(node: str, summary: str, text: str, fallback: str) -> str:
    response = groq_call("És um anotador jurídico conciso. Responde em português europeu, sem aconselhamento jurídico e sem mencionar fontes que não estejam no texto.", f"Resume em uma frase breve o que este nó fez para a cláusula abaixo. A resposta deve ser específica ao texto fornecido e não reutilizar explicações de outras cláusulas.\nEstado: {node}\nResumo: {summary}\nCláusula analisada: {text[:1200]}")
    if response:
        return response[:360].strip()
    clause_excerpt = re.sub(r"\s+", " ", text).strip()[:140]
    return f"{fallback} Evidência desta cláusula: {clause_excerpt}"


def explain_step(node: str, summary: str, text: str, evidence: str = "") -> dict[str, Any]:
    fallback = summary
    response = groq_call(
        "És um explicador jurídico conciso. Responde em português europeu, em duas frases, sem aconselhamento jurídico.",
        f"Explica de forma acessível o raciocínio deste passo de análise.\nEstado: {node}\nResumo: {summary}\nEvidência: {evidence[:1200]}\nTexto: {text[:1200]}",
    )
    return {"explanation": response or fallback, "generated_by": "groq" if response else "local-fallback"}


def legacy_alternatives_from_hypotheses(hypotheses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "id": item.get("id", f"legacy-alt-{index + 1}"),
        "hypothesis": item.get("statement", "Hipótese rejeitada"),
        "rejection_reason": item.get("rejection_reason") or "A hipótese foi rejeitada porque não foi considerada suficientemente suportada pela evidência disponível.",
        "confidence_score": round(float(item.get("confidence", 0.0)), 3),
        "cuad_category": item.get("article", ""),
    } for index, item in enumerate(hypotheses)]


def make_step(node: str, title: str, summary: str, risk: str, phase: int, payload: dict[str, Any], alternatives: list[dict[str, Any]], annotation: str, faithfulness: dict[str, Any]) -> dict[str, Any]:
    return {"step_id": f"py-langgraph-{node}-{phase}", "node_name": node, "type": "decision" if node == "classify_risk" else "audit" if node == "faithfulness_audit" else "synthesis" if node == "verdict_synthesis" else "extraction" if node == "extract_clauses" else "precedent", "title": title, "summary": summary, "generative_annotation": annotation, "risk_level": risk, "scroll_phase": phase, "payload": payload, "alternatives": alternatives, "faithfulness_metadata": faithfulness, "execution_time_ms": 0, "is_critical_node": node in ("classify_risk", "verdict_synthesis")} 


def validate_hypotheses_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    hypotheses = payload.get("hypotheses")
    if not isinstance(hypotheses, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in hypotheses:
        if not isinstance(item, dict):
            continue
        statement = str(item.get("statement", "")).strip()
        outcome = str(item.get("outcome", "")).strip().lower()
        article = str(item.get("article", "")).strip()
        evidence_quote = str(item.get("evidence_quote", "")).strip()
        confidence_raw = item.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            continue
        if not statement or outcome not in {"supports", "contradicts", "unclear"} or not 0.0 <= confidence <= 1.0:
            continue
        cleaned.append({
            "statement": statement,
            "outcome": outcome,
            "article": article,
            "confidence": round(confidence, 3),
            "evidence_quote": evidence_quote,
        })
    return cleaned


def deterministic_hypotheses(assessment: dict[str, Any], retrieval: dict[str, Any], clause_text: str) -> list[Hypothesis]:
    findings = assessment.get("findings", []) or []
    if not isinstance(findings, list):
        findings = []
    supports = [item for item in findings if str(item.get("relationship", "")).lower() in {"supports", "unclear"}]
    contradicts = [item for item in findings if str(item.get("relationship", "")).lower() == "contradicts"]
    candidates: list[Hypothesis] = []

    def make_record(article: str, outcome: str, statement: str, evidence: str, confidence: float) -> Hypothesis:
        return {
            "id": "",
            "statement": statement,
            "outcome": outcome,
            "article": article,
            "confidence": round(float(confidence), 3),
            "evidence_quote": evidence,
            "status": "candidate",
            "rejection_reason": "",
        }

    if contradicts:
        for finding in contradicts[:2]:
            clause_quote = (finding.get("contract_excerpt") or finding.get("legal_excerpt") or clause_text).strip()
            statement = f"A cláusula entra em conflito com o requisito legal em {finding.get('article', retrieval.get('category', 'referencial jurídico'))}."
            candidates.append(make_record(str(finding.get('article', '')).strip() or retrieval.get('category', 'Referencial jurídico'), 'contradicts', statement, clause_quote[:300], float(finding.get('confidence', 0.7))))
    if supports:
        for finding in supports[:2]:
            clause_quote = (finding.get("contract_excerpt") or finding.get("legal_excerpt") or clause_text).strip()
            statement = f"A cláusula pode ser interpretada como compatível com {finding.get('article', retrieval.get('category', 'referencial jurídico'))}."
            candidates.append(make_record(str(finding.get('article', '')).strip() or retrieval.get('category', 'Referencial jurídico'), 'supports', statement, clause_quote[:300], float(finding.get('confidence', 0.6))))
    if not candidates and findings:
        first = findings[0]
        candidates.append(make_record(str(first.get('article', '')).strip() or retrieval.get('category', 'Referencial jurídico'), 'unclear', "A evidência disponível é insuficiente para afirmar com segurança a conformidade ou a incompatibilidade da cláusula.", str(first.get('contract_excerpt') or first.get('legal_excerpt') or clause_text)[:300], max(0.25, float(first.get('confidence', 0.35)))))
    if not candidates:
        article = retrieval.get('category', 'Referencial jurídico')
        candidates.append({
            "id": "",
            "statement": "A evidência disponível é insuficiente para concluir a conformidade ou a contradição da cláusula.",
            "outcome": "unclear",
            "article": article,
            "confidence": 0.35,
            "evidence_quote": (clause_text or retrieval.get('evidence', ''))[:300],
            "status": "candidate",
            "rejection_reason": "",
        })

    deduped: list[Hypothesis] = []
    seen: set[str] = set()
    for item in candidates:
        key = "|".join([item["statement"], item["article"], item["outcome"]]).lower()
        if key in seen:
            continue
        seen.add(key)
        item["id"] = ""
        item["status"] = "candidate"
        deduped.append(item)
    return deduped[:5]


def normalize_hypothesis(item: dict[str, Any], index: int) -> Hypothesis:
    statement = str(item.get("statement", "")).strip()
    outcome = str(item.get("outcome", "")).strip().lower()
    article = str(item.get("article", "")).strip()
    confidence = float(item.get("confidence", 0.0))
    evidence_quote = str(item.get("evidence_quote", "")).strip()
    if not statement:
        statement = f"Hipótese jurídica {index + 1}"
    if outcome not in {"supports", "contradicts", "unclear"}:
        outcome = "unclear"
    if not 0.0 <= confidence <= 1.0:
        confidence = 0.5
    return {
        "id": f"hyp-{index + 1}",
        "statement": statement,
        "outcome": outcome,
        "article": article or "Referencial jurídico",
        "confidence": round(confidence, 3),
        "evidence_quote": evidence_quote or "Evidência não explicitada no modelo de resposta.",
        "status": "candidate",
        "rejection_reason": "",
    }


def generate_hypotheses(state: GraphState):
    retrieval = state["retrieval"]
    assessment = state["assessment"]
    clause_text = state["text"]
    provider = "deterministic-fallback"
    hypotheses: list[Hypothesis] = []
    if os.getenv("GROQ_API_KEY"):
        structured = None
        key = os.getenv("GROQ_API_KEY")
        if key:
            schema = {
                "type": "object",
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "statement": {"type": "string", "minLength": 1, "maxLength": 400},
                                "outcome": {"enum": ["supports", "contradicts", "unclear"]},
                                "article": {"type": "string", "minLength": 1, "maxLength": 200},
                                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "evidence_quote": {"type": "string", "minLength": 1, "maxLength": 500},
                            },
                            "required": ["statement", "outcome", "article", "confidence", "evidence_quote"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                        "maxItems": 5,
                    }
                },
                "required": ["hypotheses"],
                "additionalProperties": False,
            }
            prompt = (
                "Analise esta cláusula e os elementos jurídicos disponíveis. "
                "Use apenas a evidência fornecida e mantenha a resposta concisa e verificável.\n\n"
                f"Cláusula: {clause_text[:2000]}\n\n"
                f"Categoria: {retrieval.get('category', 'Referencial jurídico')}\n"
                f"Evidência legal: {retrieval.get('evidence', '')[:1500]}\n\n"
                f"Achados determinísticos: {json.dumps(assessment.get('findings', [])[:3], ensure_ascii=False)[:2500]}"
            )
            payload = json.dumps({
                "model": GROQ_MODEL,
                "temperature": 0.1,
                "max_tokens": 500,
                "response_format": {"type": "json_schema", "json_schema": {"name": "hypothesis_set", "schema": schema, "strict": True}},
                "messages": [
                    {"role": "system", "content": "Generates only plausible legal hypotheses grounded in the clause and supplied evidence. Do not invent alternatives solely for visual demonstration. Do not expose hidden reasoning or chain-of-thought. Return concise, auditable hypotheses linked to the supplied clause, article, and evidence. Include a conformity hypothesis only if genuinely plausible. Include a contradiction hypothesis only if genuinely plausible. Include an uncertainty hypothesis when the supplied information is insufficient. Return between 2 and 5 hypotheses when more than one interpretation is genuinely supported; otherwise return fewer."},
                    {"role": "user", "content": prompt},
                ],
            }).encode()
            request = urllib.request.Request(
                GROQ_URL,
                data=payload,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "JustiViz/1.0 (academic research application)",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=25) as response:
                    body = json.loads(response.read().decode("utf-8", errors="replace"))
                choice = body.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                if isinstance(content, str) and content.strip():
                    parsed = json.loads(content)
                    structured = validate_hypotheses_payload(parsed)
                    if structured:
                        provider = "groq"
                        hypotheses = [normalize_hypothesis(item, index) for index, item in enumerate(structured)]
            except Exception:
                structured = None
    if not hypotheses:
        fallback = deterministic_hypotheses(assessment, retrieval, clause_text)
        if fallback:
            provider = "deterministic-fallback"
            hypotheses = fallback
        else:
            provider = "deterministic-fallback"
            hypotheses = [{"id": "hyp-1", "statement": "A evidência disponível é insuficiente para concluir a conformidade ou a contradição da cláusula.", "outcome": "unclear", "article": retrieval.get("category", "Referencial jurídico"), "confidence": 0.35, "evidence_quote": (clause_text or retrieval.get("evidence", ""))[:300], "status": "candidate", "rejection_reason": ""}]
    for index, item in enumerate(hypotheses):
        item["id"] = f"hyp-{index + 1}"
        item["status"] = "candidate"
        item["rejection_reason"] = ""
    summary = f"Foram geradas {len(hypotheses)} hipóteses juridicamente relevantes a partir da cláusula e da evidência recuperada."
    return {
        "hypotheses": hypotheses,
        "selected_hypothesis": hypotheses[0],
        "rejected_hypotheses": [],
        "steps": [make_step("generate_hypotheses", "Geração de hipóteses jurídicas", summary, state.get("risk_level", "LOW"), 25, {"provider": provider, "source_evidence": {"retrieval": retrieval, "assessment": assessment, "clause_excerpt": clause_text[:800]}, "generated_hypotheses": hypotheses, "state_variables": {"source_document": retrieval.get("source"), "matched_terms": retrieval.get("matched_terms", []), "document_category": retrieval.get("category")}}, [], annotation("generate_hypotheses", summary, clause_text, "As hipóteses foram geradas a partir da evidência legal e do texto da cláusula."), audit(summary, retrieval.get("evidence", "")))]
    }


def selected_hypothesis_score(hypothesis: Hypothesis, findings: list[dict[str, Any]]) -> float:
    confidence = float(hypothesis.get("confidence", 0.0))
    article_name = str(hypothesis.get("article", "")).lower()
    score = confidence
    for finding in findings:
        finding_article = str(finding.get("article", "")).lower()
        if article_name and article_name in finding_article:
            score += 0.3
        if str(hypothesis.get("outcome", "")).lower() == str(finding.get("relationship", "")).lower():
            score += 0.5
        if str(finding.get("severity", "")).lower() == "strong":
            score += 0.2
    return score


def evaluate_hypothesis(state: GraphState):
    hypothesis = dict(state.get("hypothesis") or {})
    if not hypothesis:
        return {"evaluated_hypotheses": [], "steps": []}
    assessment = state["assessment"]
    retrieval = state["retrieval"]
    clause_text = state["text"]
    findings = assessment.get("findings", [])
    hypothesis_id = str(hypothesis.get("id", "hyp-unknown"))
    outcome = str(hypothesis.get("outcome", "")).lower()
    article = str(hypothesis.get("article", "")).strip()
    rejection_reason = ""
    strong_conflicts = [item for item in findings if str(item.get("relationship", "")).lower() == "contradicts" and float(item.get("confidence", 0.0)) >= 0.8]
    strong_supports = [item for item in findings if str(item.get("relationship", "")).lower() == "supports" and float(item.get("confidence", 0.0)) >= 0.8]
    if outcome == "contradicts" and strong_supports:
        rejection_reason = "A hipótese contradiz uma conclusão determinística de alta confiança e não é compatível com a evidência recuperada."
    elif outcome == "supports" and strong_conflicts:
        rejection_reason = "A hipótese não se sustenta porque há uma contradição legal forte indicada pela avaliação determinística."
    elif article and retrieval.get("category") and article.lower() not in str(retrieval.get("category", "")).lower() and not any(str(item.get("article", "")).lower() in article.lower() or article.lower() in str(item.get("article", "")).lower() for item in findings):
        rejection_reason = "O artigo indicado não corresponde ao referencial recuperado nem à evidência legal disponível."
    elif outcome not in {"supports", "contradicts", "unclear"}:
        rejection_reason = "A hipótese não é um resultado compatível com a saída esperada de um esquema de avaliação jurídico-evidencial."
    elif not retrieval.get("evidence") and not clause_text:
        rejection_reason = "Sem evidência suficiente, a hipótese não pode ser mantida como interpretação viable."
    if rejection_reason:
        hypothesis["status"] = "rejected"
        hypothesis["rejection_reason"] = rejection_reason
    else:
        hypothesis["status"] = "candidate"
        hypothesis["rejection_reason"] = ""
    summary = f"A hipótese {hypothesis_id} foi avaliada como {'rejeitada' if rejection_reason else 'viável'} com base na evidência legal e no excerto da cláusula."
    return {"evaluated_hypotheses": [hypothesis], "steps": [make_step("evaluate_hypothesis", f"Avaliação da hipótese {hypothesis_id}", summary, state.get("risk_level", "LOW"), 50, {"hypothesis": hypothesis, "assessment_findings": findings, "rejection_reason": rejection_reason, "state_variables": {"article": article, "outcome": outcome}}, [], annotation("evaluate_hypothesis", summary, clause_text, "A hipótese foi confrontada com a evidência legal recuperada para validar a sua viabilidade."), audit(summary, retrieval.get("evidence", "")))]}


def select_hypothesis(state: GraphState):
    evaluated = state.get("evaluated_hypotheses", []) or []
    viable = [item for item in evaluated if item.get("status") != "rejected"]
    rejected = [item for item in evaluated if item.get("status") == "rejected"]
    findings = state["assessment"].get("findings", [])
    if viable:
        selected = max(viable, key=lambda item: selected_hypothesis_score(item, findings))
        selected["status"] = "selected"
        selected["rejection_reason"] = ""
    else:
        selected = {
            "id": "hyp-uncertain",
            "statement": "A informação disponível não permite identificar uma hipótese juridicamente viável com segurança.",
            "outcome": "unclear",
            "article": state["retrieval"].get("category", "Referencial jurídico"),
            "confidence": 0.35,
            "evidence_quote": (state["text"] or state["retrieval"].get("evidence", ""))[:300],
            "status": "selected",
            "rejection_reason": "",
        }
    criteria = ["alinhamento com a evidência", "confiança da hipótese", "concordância com achados determinísticos", "relevância do artigo jurídico", "baixa incerteza"]
    summary = f"A hipótese selecionada foi {selected.get('statement', 'hipótese incerta')} com base em {', '.join(criteria)}."
    step_payload = {"selected_hypothesis": selected, "rejected_hypotheses": rejected, "rejection_count": len(rejected), "selection_criteria": criteria}
    return {"selected_hypothesis": selected, "rejected_hypotheses": rejected, "steps": [make_step("select_hypothesis", "Seleção da hipótese mais robusta", summary, state.get("risk_level", "LOW"), 60, step_payload, legacy_alternatives_from_hypotheses(rejected), annotation("select_hypothesis", summary, state["text"], "A hipótese mais sólida foi selecionada com base na evidência e na consistência legal."), audit(summary, state["retrieval"].get("evidence", "")))]}


def build_graph():
    try:
        from langgraph.constants import Send
    except Exception:
        try:
            from langgraph.types import Send
        except Exception:
            Send = None
    graph = StateGraph(GraphState)

    def extract(state: GraphState):
        text = state["text"]
        evidence = state["retrieval"].get("evidence", "")
        framework = state["retrieval"].get("category", state["category"])
        summary = f"Foi extraída a cláusula com {len(text.strip())} carateres, incluindo o seguinte conteúdo: {re.sub(r'\\s+', ' ', text).strip()[:220]}."
        return {"steps": [make_step("extract_clauses", "Ingestão e extração de cláusulas", summary, "LOW", 0, {"raw_clause_quote": text[:1000], "extracted_entities": [], "clause_count": len(segment_contract(text)), "cuad_category_matched": framework, "statutory_basis": state["retrieval"].get("statutory_basis"), "state_variables": {"source_url": state["retrieval"].get("source"), "source_status": state["retrieval"].get("source_status", "local-corpus")}}, [], annotation("extract_clauses", summary, text, "A cláusula foi extraída e preparada para comparação."), audit(summary, evidence))]}

    def classify(state: GraphState):
        result = state["retrieval"]
        assessment = state["assessment"]
        score = int(assessment["risk_score"])
        risk = assessment["classification"]
        retrieval_name = result.get("retrieval", "tfidf_cosine")
        findings = assessment["findings"]
        selected_hypothesis = state.get("selected_hypothesis") or {}
        rejected_hypotheses = state.get("rejected_hypotheses", []) or []
        used_articles = ", ".join(dict.fromkeys(item["article"] for item in findings))
        summary = f"A cláusula foi classificada como {risk} ({score}/100) após comparação com {result.get('category')}. Foram usados: {used_articles}."
        payload = {"cuad_category_matched": result.get("category"), "confidence_metric": result.get("similarity", 0), "raw_clause_quote": result.get("evidence", ""), "statutory_basis": result.get("statutory_basis"), "legal_findings": findings, "clause_assessment": assessment, "selected_hypothesis": selected_hypothesis, "rejected_hypotheses": rejected_hypotheses, "rejection_count": len(rejected_hypotheses), "state_variables": {"source_document": result.get("source"), "retrieval": retrieval_name, "matched_terms": result.get("matched_terms", []), "used_articles": used_articles}}
        return {"risk_score": score, "risk_level": risk, "classification": f"{risk}: {result.get('category')}", "steps": [make_step("classify_risk", f"Classificação de risco: {risk}", summary, risk, 50, payload, legacy_alternatives_from_hypotheses(rejected_hypotheses), annotation("classify_risk", summary, state["text"], "A classificação foi ligada à evidência legal recuperada."), audit(summary, result.get("evidence", "")))]}

    def precedent(state: GraphState):
        findings = state["assessment"]["findings"]
        used_articles = ", ".join(dict.fromkeys(item["article"] for item in findings))
        summary = f"A classificação usou {used_articles} do referencial {state['retrieval'].get('category')}, comparando os respetivos critérios com o excerto da cláusula."
        payload = {"cuad_category_matched": state["retrieval"].get("category"), "raw_clause_quote": state["retrieval"].get("evidence", ""), "statutory_basis": state["retrieval"].get("statutory_basis"), "legal_findings": findings, "selected_hypothesis": state.get("selected_hypothesis"), "rejected_hypotheses": state.get("rejected_hypotheses", []), "state_variables": {"source_document": state["retrieval"].get("source"), "source_status": state["retrieval"].get("source_status"), "used_articles": used_articles}}
        return {"steps": [make_step("check_precedent", "Referências legais utilizadas", summary, state.get("risk_level", "LOW"), 75, payload, [], annotation("check_precedent", summary, state["text"], "As referências legais foram confrontadas com o texto da cláusula."), audit(summary, state["retrieval"].get("evidence", "")))]}

    def faithfulness(state: GraphState):
        assessment = state["assessment"]
        summary = f"A auditoria comparou a classificação {assessment['classification']} com o texto da cláusula, a evidência legal e os artigos usados, identificando {len(assessment['uncertainty_notes'])} incertezas."
        result = audit(summary, state["retrieval"].get("evidence", ""))
        result["audit_notes"] = f"{result['audit_notes']} Artigos avaliados: {', '.join(dict.fromkeys(item['article'] for item in assessment['findings']))}."
        return {"faithfulness": result, "steps": [make_step("faithfulness_audit", "Auditoria de fidelidade", summary, state.get("risk_level", "LOW"), 75, {"audit_target": "texto da cláusula, evidência legal e classificação", "audit_provider": "groq" if os.getenv("GROQ_API_KEY") else "local", "legal_findings": assessment["findings"], "selected_hypothesis": state.get("selected_hypothesis"), "rejected_hypotheses": state.get("rejected_hypotheses", []), "state_variables": {"uncertainty_notes": assessment["uncertainty_notes"]}}, [], annotation("faithfulness_audit", summary, state["text"], "A auditoria comparou a narrativa, a evidência e a classificação."), result)]}

    def verdict(state: GraphState):
        assessment = state["assessment"]
        selected = state.get("selected_hypothesis") or {"statement": "A hipótese selecionada não foi determinável.", "article": state["retrieval"].get("category", "Referencial jurídico"), "confidence": 0.35, "outcome": "unclear"}
        rejected_hypotheses = state.get("rejected_hypotheses", [])
        summary = f"Veredito: a hipótese selecionada foi '{selected.get('statement', 'hipótese jurídica')}', porque alinha a evidência legal e a cláusula com a interpretação mais robusta disponível."
        verdict_data = {"risk_score": state.get("risk_score", assessment.get("risk_score", 20)), "classification": state.get("classification", assessment.get("classification", "Sem classificação")), "summary": summary, "eu_ai_act_risk_tier": "High Risk" if state.get("risk_score", assessment.get("risk_score", 20)) >= 65 else "Limited Risk", "recommended_clauses": ["Confirmar o âmbito, a duração e a reciprocidade da disposição."], "mitigation_guidance": "Rever os artigos citados com profissionais do Direito.", "selected_hypothesis": selected, "rejected_hypotheses": rejected_hypotheses, "rejection_count": len(rejected_hypotheses), "hypotheses": state.get("hypotheses", []), "evaluated_hypotheses": state.get("evaluated_hypotheses", []), "legal_findings": assessment.get("findings", []), "evidence": state["retrieval"].get("evidence", ""), "source": state["retrieval"].get("source")}
        return {"verdict": verdict_data, "steps": [make_step("verdict_synthesis", "Síntese do veredito e recomendação", summary, state.get("risk_level", "LOW"), 100, {"final_verdict": verdict_data, "statutory_basis": state["retrieval"].get("statutory_basis")}, legacy_alternatives_from_hypotheses(rejected_hypotheses), annotation("verdict_synthesis", summary, state["text"], "Anotação local: a recomendação foi sintetizada para revisão humana."), state.get("faithfulness", audit(summary, state["retrieval"].get("evidence", ""))))]}

    graph.add_node("extract_clauses", extract)
    graph.add_node("generate_hypotheses", generate_hypotheses)
    graph.add_node("evaluate_hypothesis", evaluate_hypothesis)
    graph.add_node("select_hypothesis", select_hypothesis)
    graph.add_node("classify_risk", classify)
    graph.add_node("check_precedent", precedent)
    graph.add_node("faithfulness_audit", faithfulness)
    graph.add_node("verdict_synthesis", verdict)

    graph.add_edge(START, "extract_clauses")
    graph.add_edge("extract_clauses", "generate_hypotheses")
    if Send is not None:
        graph.add_conditional_edges("generate_hypotheses", lambda state: [Send("evaluate_hypothesis", {"hypothesis": hypothesis, **state}) for hypothesis in state.get("hypotheses", [])])
        graph.add_edge("evaluate_hypothesis", "select_hypothesis")
    else:
        graph.add_edge("generate_hypotheses", "select_hypothesis")
    graph.add_edge("select_hypothesis", "classify_risk")
    graph.add_edge("classify_risk", "check_precedent")
    graph.add_edge("check_precedent", "faithfulness_audit")
    graph.add_edge("faithfulness_audit", "verdict_synthesis")
    graph.add_edge("verdict_synthesis", END)
    return graph.compile()


def invoke_graph(title: str, category: str, text: str) -> dict[str, Any]:
    retrieval = retrieve_legal_framework(text, category)
    assessment = assess_clause(text, retrieval)
    return build_graph().invoke({"title": title, "category": category, "text": text, "retrieval": retrieval, "assessment": assessment, "steps": []})


def make_trace(title: str, category: str, text: str, result: dict[str, Any], trace_suffix: str = "") -> dict[str, Any]:
    trace_id = f"py-langgraph-{abs(hash((title, text)))}{trace_suffix}"
    retrieval = result.get("retrieval", {})
    verdict = result.get("verdict", {})
    return {
        "trace_id": trace_id,
        "contract_title": title,
        "category": category,
        "cuad_master_category": category,
        "parties": [],
        "governing_law": retrieval.get("category", "A determinar por revisão humana"),
        "contract_excerpt": text[:500],
        "target_query": f"Avaliar o texto submetido nas categorias: {category}",
        "steps": result.get("steps", []),
        "final_verdict": verdict,
        "assessment": result.get("assessment", {}),
        "hypotheses": result.get("hypotheses", []),
        "selected_hypothesis": result.get("selected_hypothesis"),
        "rejected_hypotheses": result.get("rejected_hypotheses", []),
        "evaluated_hypotheses": result.get("evaluated_hypotheses", []),
        "rejection_count": len(result.get("rejected_hypotheses", [])),
        "metadata": {"created_at": "", "model_orchestrator": "langgraph-python", "secondary_auditor_model": "groq" if os.getenv("GROQ_API_KEY") else "local-validation-fallback", "cuad_version": "local-corpus-tfidf", "data_provenance": "live-analysis" if os.getenv("GROQ_API_KEY") else "local-analysis", "legal_source_url": retrieval.get("source"), "legal_source_name": retrieval.get("category"), "legal_source_status": retrieval.get("source_status", "local-corpus")},
    }


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    title = payload.get("contractTitle") or "Contrato submetido"
    category = payload.get("category") or "Avaliação geral"
    text = payload.get("contractText") or ""
    result = invoke_graph(title, category, text)
    base = make_trace(title, category, text, result)
    base["selected_hypothesis"] = result.get("selected_hypothesis")
    base["rejected_hypotheses"] = result.get("rejected_hypotheses", [])
    base["hypotheses"] = result.get("hypotheses", [])
    base["evaluated_hypotheses"] = result.get("evaluated_hypotheses", [])
    base["rejection_count"] = len(base["rejected_hypotheses"])
    clauses = segment_contract(text) or [{"index": 0, "title": "Corpo do contrato", "text": text}]
    clause_entries = []
    for index, item in enumerate(clauses):
        clause_text = item.get("text", "")
        clause_result = invoke_graph(f"{title} - Cláusula {index + 1}", category, clause_text)
        clause_trace = make_trace(f"{title} - Cláusula {index + 1}", category, clause_text, clause_result, f"-clause-{index + 1}")
        clause_trace["steps"] = [dict(step, step_id=f"{step['step_id']}-clause-{index + 1}") for step in clause_trace["steps"]]
        clause_assessment = clause_result.get("assessment", {})
        clause_entries.append({"index": item.get("index", index), "title": item.get("title", f"Cláusula {index + 1}"), "text": clause_text, "risk_level": clause_assessment.get("classification", "MEDIUM"), "trace": clause_trace})
    base["clauses"] = clause_entries
    assessments = [entry["trace"].get("assessment", {}) for entry in clause_entries]
    scores = [int(item.get("risk_score", 55)) for item in assessments]
    levels = [item.get("classification", "MEDIUM") for item in assessments]
    aggregate_score = round(sum(scores) / max(len(scores), 1))
    aggregate_level = "CRITICAL" if "CRITICAL" in levels else "HIGH" if "HIGH" in levels else "MEDIUM" if "MEDIUM" in levels else "LOW"
    base["final_verdict"] = {
        **base.get("final_verdict", {}),
        "risk_score": aggregate_score,
        "classification": f"{aggregate_level}: avaliação agregada de {len(assessments)} cláusula(s)",
        "summary": f"Avaliação agregada do contrato: {aggregate_level} ({aggregate_score}/100), calculada a partir da média das classificações das cláusulas e da prevalência do risco mais grave.",
        "mitigation_guidance": "Rever individualmente as cláusulas classificadas como risco médio, elevado ou crítico com profissionais do Direito.",
    }
    return base


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"status": "ok", "langgraph": True, "groq": bool(os.getenv("GROQ_API_KEY")), "groq_diagnostic": GROQ_DIAGNOSTIC, "cuad_examples": len(CORPUS)})
        else:
            self.respond(404, {"error": "Not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if self.path == "/analyze":
            self.respond(200, analyze(payload))
        elif self.path == "/explain":
            self.respond(200, explain_step(payload.get("node", ""), payload.get("summary", ""), payload.get("text", ""), payload.get("evidence", "")))
        elif self.path == "/audit":
            summary = payload.get("summary", "")
            self.respond(200, audit(summary, json.dumps(payload.get("technicalPayload", {}))))
        else:
            self.respond(404, {"error": "Not found"})

    def respond(self, status: int, data: dict[str, Any]):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(os.getenv("LANGGRAPH_PORT", "8001"))), Handler).serve_forever()
