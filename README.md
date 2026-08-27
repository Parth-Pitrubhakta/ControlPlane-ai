# ControlPlane.ai

> **Risk-Adaptive AI Oversight for Performance, Cost & Responsibility**

ControlPlane.ai is a **model-agnostic control layer for AI applications and agents** that continuously evaluates AI responses across three critical dimensions:

* **Performance** — Is the response correct, factual, and supported by evidence?
* **Cost** — Is the system using more tokens, tools, latency, or compute than necessary?
* **Responsibility** — Is the response biased, unsafe, non-compliant, or exposing sensitive information?

Instead of applying expensive verification to every AI response, ControlPlane.ai estimates risk first and dynamically chooses the appropriate verification depth.

---

## 🚨 The Problem

Enterprise AI systems can fail silently after deployment.

An AI response may be:

* Confidently incorrect or hallucinated
* Unsupported by evidence
* Unnecessarily expensive
* Biased or unsafe
* Non-compliant with organizational policies
* Exposing personally identifiable or sensitive information

These problems are often discovered **only after a user has already acted on the output**.

The core challenge is:

> **How can we catch risky AI responses first without adding so much latency that oversight defeats the purpose?**

---

## 💡 Our Solution

ControlPlane.ai sits between an AI application and the model/agent infrastructure as a **risk-adaptive oversight layer**.

```text
                    ┌─────────────────┐
                    │    User / App   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   AI Gateway    │
                    │ Prompt + Meta   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   AI / Agent    │
                    │  Any Model      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Risk Router     │
                    │ Predict Risk    │
                    └────────┬────────┘
                             │
             ┌───────────────┼────────────────┐
             ▼               ▼                ▼
       ┌───────────┐   ┌───────────┐   ┌──────────────┐
       │Performance│   │   Cost    │   │Responsibility│
       │   Guard   │   │   Guard   │   │    Guard     │
       └─────┬─────┘   └─────┬─────┘   └──────┬───────┘
             │               │                │
             └───────────────┼────────────────┘
                             ▼
                    ┌─────────────────┐
                    │ Decision Engine │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
            ALLOW           EDIT           BLOCK
                                            │
                                            ▼
                                      HUMAN REVIEW
```

---

## 🧠 Core Architecture

The ControlPlane pipeline consists of six major stages:

### 1. Request

The user or application sends a request to the AI system.

### 2. AI Gateway

The gateway captures the prompt, response metadata, model information, and relevant execution information.

### 3. AI / Agent

The request is processed by the selected AI model or agent.

ControlPlane.ai is designed to remain **model-agnostic**.

### 4. Response

The system captures the generated response along with relevant telemetry such as:

* Token usage
* Latency
* Tool calls
* Model information
* Execution metadata

### 5. Risk Router

The Risk Router estimates the risk associated with the response and determines how much verification is required.

### 6. Decision

The system chooses an appropriate intervention:

* **ALLOW** — Response is considered sufficiently safe/reliable.
* **EDIT** — Response should be modified before delivery.
* **BLOCK** — Response should not reach the user.
* **HUMAN** — Escalate the response for human review.

---

# 🛡️ Three Risk Detection Guards

## A. Performance Guard

The Performance Guard evaluates whether the AI response is reliable and factually supported.

Potential checks include:

* Hallucination detection
* Factuality verification
* Claim verification
* Contradiction detection
* Citation verification
* Confidence estimation

Potential evaluation datasets:

* RAGTruth
* HaluEval

---

## B. Cost Guard

The Cost Guard evaluates whether the AI system is using resources efficiently.

Potential signals include:

* Token usage
* Tool-call frequency
* Response latency
* Model selection
* Cost anomalies
* Serving traces
* Compute usage

The goal is not simply to identify expensive responses, but to determine whether the cost is **appropriate for the task and outcome**.

---

## C. Responsibility Guard

The Responsibility Guard evaluates safety, fairness, privacy, and policy compliance.

Potential checks include:

* Bias detection
* Harmful-content detection
* Safety classification
* PII detection
* Data-leakage detection
* Policy compliance
* Organizational rule enforcement

Potential evaluation datasets include BBQ for bias evaluation.

---

# ⚡ Risk-Adaptive Verification

The key differentiator of ControlPlane.ai is **risk-adaptive verification**.

The system does not apply the same expensive verification pipeline to every response.

Instead:

```text
                  AI Response
                       │
                       ▼
                ┌─────────────┐
                │ Risk Score  │
                └──────┬──────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       LOW RISK    MEDIUM RISK   HIGH RISK
          │            │            │
          ▼            ▼            ▼
      FAST CHECK    DEEP CHECK   FULL CHECK
                                    +
                              HUMAN ESCALATION
```

### 🟢 Low Risk

Use lightweight validation for routine responses.

### 🟡 Medium Risk

Perform additional factuality and responsibility checks.

### 🔴 High Risk

Perform comprehensive multi-dimensional validation and potentially escalate to human review.

This allows ControlPlane.ai to balance:

**Reliability ↔ Latency ↔ Cost**

---

# 📊 Explainable Risk Reports

ControlPlane.ai should not only produce a risk score.

It should explain **why** a response was considered risky.

Example:

```text
CONTROLPLANE REPORT

AI Response:
"The product has a 5-year warranty."

PERFORMANCE       ⚠ HIGH
No supporting evidence found.
Policy documentation states 3 years.

COST              ✓ LOW
Expected token and tool usage.

RESPONSIBILITY    ✓ LOW
No bias, safety, or privacy issue detected.

OVERALL RISK      HIGH

ACTION            → EDIT / BLOCK

RECOMMENDATION:
Replace the unsupported claim with a verified statement.
```

The goal is to make every intervention **traceable and explainable**.

---

# 🎯 Success Metrics

ControlPlane.ai can be evaluated using:

* Hallucination detection accuracy
* Factuality detection accuracy
* Bias detection performance
* Safety detection performance
* PII leakage detection
* Cost anomaly detection
* Verification latency
* Verification overhead
* False-positive rate
* False-negative rate
* Percentage of risky responses caught before delivery

---

# 🏗️ Project Structure

The repository is expected to evolve toward a modular architecture similar to:

```text
controlplane-ai/
│
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── .gitignore
├── .env.example
│
├── docs/
│   ├── architecture/
│   ├── api/
│   └── evaluation/
│
├── src/
│   └── controlplane/
│       ├── gateway/
│       ├── router/
│       ├── guards/
│       │   ├── performance/
│       │   ├── cost/
│       │   └── responsibility/
│       ├── decision/
│       ├── telemetry/
│       └── common/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/
│
├── examples/
│
├── scripts/
│
└── configs/
```

The exact structure may change as implementation progresses.

---

# 🤝 Contributing

ControlPlane.ai is developed using a **pull-request based workflow**.

We encourage every team member to work independently through feature branches and submit changes for review.

### Development workflow

```text
main
 │
 ├── feature/performance-guard
 │
 ├── feature/cost-guard
 │
 ├── feature/responsibility-guard
 │
 └── feature/risk-router
```

Each contributor should:

1. Create a branch from `main`.
2. Implement their feature.
3. Test their changes.
4. Commit the changes.
5. Push the branch.
6. Open a Pull Request.
7. Request review.
8. Address review comments.
9. Get approval.
10. Merge into `main`.

Direct pushes to `main` should be avoided.

---

# 🌿 Branch Naming

Use descriptive branch names.

```text
feature/<feature-name>
fix/<bug-name>
refactor/<component-name>
docs/<documentation-name>
test/<test-name>
experiment/<experiment-name>
```

Examples:

```text
feature/performance-guard
feature/risk-router
feature/pii-detector
fix/token-calculation
test/hallucination-evaluation
docs/architecture
```

---

# 📝 Commit Convention

Use clear, descriptive commits.

Examples:

```text
feat: add performance risk scoring
feat: implement cost anomaly detector
fix: handle missing token metadata
test: add hallucination guard tests
refactor: simplify risk router
docs: update architecture documentation
```

Recommended format:

```text
<type>: <short description>
```

---

# 🔍 Pull Requests

Every Pull Request should explain:

### What changed?

Briefly describe the implementation.

### Why?

Explain the problem the change solves.

### How was it tested?

Include relevant tests, benchmarks, or evaluation results.

### Example

```text
## What changed

Implemented the initial Performance Guard.

## Why

We need to identify responses that contain unsupported factual claims.

## Testing

- Added unit tests for claim extraction
- Tested against sample RAG responses
- Added factuality scoring tests

## Related Issue

Closes #12
```

---

# 🔐 Repository Rules

The `main` branch should be protected.

Recommended rules:

* No direct pushes to `main`
* Pull Request required
* At least **1 approval** required
* CI checks must pass before merging
* Branch must be up to date before merging
* Contributors cannot approve their own Pull Requests
* Stale approvals should be dismissed after significant changes

The repository maintainer/reviewer has final merge authority.

---

# 🧪 Testing

Every feature should include appropriate tests.

```text
tests/
├── unit/
├── integration/
└── evaluation/
```

Before opening a Pull Request, contributors should ensure:

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run formatting
ruff format .
```

Commands may change depending on the final technology stack.

---

# 🔬 Evaluation

Because ControlPlane.ai is an AI oversight system, evaluation is a first-class part of development.

Each detection component should ideally be evaluated using:

* Accuracy
* Precision
* Recall
* F1 score
* False-positive rate
* False-negative rate
* Latency
* Computational overhead

For risk-adaptive routing, we additionally care about:

* Percentage of risky responses detected
* Average verification cost
* Average verification latency
* Detection coverage
* Escalation rate

---

# 🚀 Roadmap

Potential development stages:

### Phase 1 — Foundation

* [ ] Project scaffolding
* [ ] AI Gateway
* [ ] Common response schema
* [ ] Basic telemetry
* [ ] Risk scoring interface

### Phase 2 — Risk Detection

* [ ] Performance Guard
* [ ] Cost Guard
* [ ] Responsibility Guard
* [ ] Unified risk score

### Phase 3 — Risk Router

* [ ] Low-risk fast path
* [ ] Medium-risk deep verification
* [ ] High-risk full verification
* [ ] Human escalation

### Phase 4 — Intervention

* [ ] Allow
* [ ] Edit
* [ ] Block
* [ ] Human review

### Phase 5 — Explainability

* [ ] Risk reports
* [ ] Evidence tracking
* [ ] Decision explanations
* [ ] Audit logs

### Phase 6 — Evaluation & Deployment

* [ ] Benchmark suite
* [ ] Performance testing
* [ ] Integration tests
* [ ] API
* [ ] Dashboard
* [ ] Production deployment

---

# 📄 License

This project is currently under development.

License information will be added as the project matures.

---

## ⭐ Vision

ControlPlane.ai aims to move enterprise AI oversight from:

> **"We discovered the problem after the AI was used."**

to:

> **"We detected the risk before the AI response reached the user."**

The long-term goal is to provide an intelligent control layer that makes AI systems **more reliable, cost-efficient, safe, and accountable** without imposing unnecessary verification overhead.
