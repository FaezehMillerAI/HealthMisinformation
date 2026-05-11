# Explainability-Enhanced Methodology for Financial Misinformation Detection with GPT-4o-mini-IT

## 1. Overview

The Financial Misinformation Detection (FMD) shared task evaluates systems that classify financial claims as True, False, or Not Enough Information (NEI) and generate free-text explanations grounded in contextual justifications using the FIN-FACT dataset. The referenced system instruction-tunes GPT-4o-mini (GPT-4o-mini-IT) to jointly output a label and a short evidence-style explanation, achieving strong F1 and ROUGE scores but relying solely on unstructured natural language rationales. This report proposes an extended, more faithful and auditable explainability methodology that augments GPT-4o-mini-IT with structured rationales, span-level attributions, and counterfactual reasoning while remaining compatible with the existing FMD setting.[^1]

## 2. Current Approach and Explainability Limitations

The current methodology concatenates a system prompt, an updated claim (claim + sci-digest), and the context as input, and trains GPT-4o-mini-IT to output both the label and free-text evidence text that mimics FIN-FACT explanations. Explanations are evaluated with ROUGE-1/2/L against gold evidence, and the overall leaderboard score is the average of micro-F1 (labels) and ROUGE-1. This yields explanations that are linguistically similar to ground truth but leaves several explainability gaps:[^1]

1. **Unstructured rationale**: Explanations are plain text with no explicit links to specific context sentences or spans, making it difficult to verify which parts of the context actually drive the prediction.[^1]
2. **Faithfulness not enforced**: The model may generate plausible-sounding rationales that are only loosely connected to the internal decision process (“self-rationalization”), and there is no explicit training signal to ensure explanations are necessary and sufficient for the label.
3. **Single-level granularity**: There is no sentence- or token-level attribution, no structured decomposition of reasoning steps, and no notion of uncertainty or alternative plausible labels.
4. **Limited use of FIN-FACT evidence**: FIN-FACT provides detailed evidence texts but these are used only as targets for generation, not as supervision for extracting minimal decision-critical spans from the context.[^1]

Addressing these limitations requires a methodology that ties explanations more tightly to the model’s decision-making while remaining practical within the constraints of the FMD task.

## 3. Proposed Framework: Counterfactually-Grounded Structured Explanations (CGSE)

The proposed methodology, Counterfactually-Grounded Structured Explanations (CGSE), augments GPT-4o-mini-IT with an additional structured explainability layer and training objectives that explicitly encourage faithful, span-grounded, and counterfactually robust explanations.

At a high level, each prediction is represented as a tuple:

- Predicted label \( \hat{y} \in \{\text{True}, \text{False}, \text{NEI}\} \)
- Rationale spans \( R \) (one or more contiguous token or sentence spans from the context)
- Structured reasoning summary \( S \) (a concise, templated explanation conditioned on \( R \))
- Counterfactual variant \( (\tilde{x}, \tilde{y}) \) illustrating how changing key evidence would flip or preserve the label

GPT-4o-mini-IT remains the core sequence model, but explanation generation is decomposed into evidence selection, structured reasoning, and counterfactual assessment phases.

## 4. Evidence Alignment and Rationale Supervision

### 4.1 Using FIN-FACT Evidence as Weak Rationales

FIN-FACT includes an `evidence` field providing human-written explanations justifying each label. These explanations can be exploited as weak supervision for span-level rationales by aligning them to sentences or substrings in the context:[^1]

- Split the context into sentences and compute semantic similarity between each sentence and the evidence text using a sentence encoder (e.g., a small transformer or embeddings derived from GPT-4o-mini).[^1]
- Mark the top-k sentences with highest similarity as candidate rationales, optionally refining with token-level overlap to capture finer spans.
- Treat these selected sentences/spans as pseudo-gold rationales \( R^* \) for training a rationale extractor, while still using the full original FIN-FACT `evidence` field as the target for free-text explanation generation.

This step converts the dataset from label + explanation-only supervision to label + explanation + span-level rationale supervision without additional human annotation.

### 4.2 Rationale Extraction Module

The extended model incorporates a lightweight rationale extractor on top of GPT-4o-mini-IT’s contextual representations:

- Encode the claim and context with GPT-4o-mini-IT as usual.
- Add a token- or sentence-level classification head that outputs a Bernoulli decision for whether each token/sentence belongs to the rationale.
- Train this head using binary cross-entropy loss against the pseudo-gold rationale mask derived from Section 4.1.

This produces explicit rationale spans \( R \) that can be surfaced to users (e.g., via highlighting in the context) and used to constrain downstream explanation generation.

## 5. Faithfulness via Sufficiency and Comprehensiveness Objectives

To make rationales faithful rather than purely descriptive, the methodology adds sufficiency and comprehensiveness constraints inspired by selective rationalization work.

1. **Sufficiency**: The model should correctly predict the label when given only the selected rationale spans \( R \).
   - Construct an input where the context is replaced by only the selected rationale sentences.
   - Apply the classification head to this reduced input and encourage the predicted \( \hat{y}_{\text{rationale}} \) to match the original label \( y \) via cross-entropy.

2. **Comprehensiveness**: Removing the rationale from the context should significantly degrade confidence in the original label.
   - Construct an input where the rationale spans are masked or removed from the context.
   - Encourage the model’s probability for the original label to drop relative to the full-context input, e.g., by maximizing the difference in log-probability between original and rationale-removed variants.

These objectives are integrated into the overall loss:

\[
\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{label}} + \lambda_{\text{rat}} \mathcal{L}_{\text{rationale}} + \lambda_{\text{suff}} \mathcal{L}_{\text{suff}} + \lambda_{\text{comp}} \mathcal{L}_{\text{comp}} + \lambda_{\text{exp}} \mathcal{L}_{\text{explanation}}.
\]

Here, \( \mathcal{L}_{\text{label}} \) is the original classification loss, \( \mathcal{L}_{\text{rationale}} \) is the span prediction loss, \( \mathcal{L}_{\text{suff}} \) and \( \mathcal{L}_{\text{comp}} \) enforce faithfulness, and \( \mathcal{L}_{\text{explanation}} \) remains the sequence-to-sequence loss for generating FIN-FACT-style evidence text.

## 6. Structured Explanation Generation

With rationales \( R \) available, explanations are generated in a more structured manner rather than as unconstrained free text.

### 6.1 Explanation Template Schema

A simple schema for explanations might be:

1. Identify key financial entities and variables (company names, instruments, time frames, numerical values) in the claim.
2. State the relationship asserted by the claim in a normalized form.
3. Indicate the rationale sentences that confirm, contradict, or fail to address the assertion.
4. Map this to one of the labels (True, False, NEI) with an explicit rule.

GPT-4o-mini-IT can be instructed, via a refined system prompt, to emit explanations in a semi-structured, but still natural, style such as:

- "The context explicitly states [summary of rationale], which directly confirms/contradicts the claim that [normalized claim]. Therefore, the claim is [label]."

During training, the model conditions explanation generation on the selected rationale spans by either:

- Prefixing the explanation decoder input with a serialized view of \( R \) (e.g., concatenating rationale sentences before the full context), or
- Using attention masking to focus the decoder primarily on rationale tokens.

### 6.2 Explanation–Label Consistency Regularization

To further couple explanations with labels, an auxiliary classifier is trained to predict the label from the generated explanation alone. A consistency loss encourages the label inferred from the explanation to match the primary label, reducing the risk of fluent but label-inconsistent rationales.

## 7. Counterfactual Explanations for Financial Claims

Financial misinformation often hinges on subtle changes in entities, dates, or magnitudes (e.g., swapping quarterly vs annual results, or misstating percentage changes). CGSE introduces a counterfactual explanation component to make such sensitivities explicit.[^1]

### 7.1 Counterfactual Data Generation

Synthetic counterfactual variants \( (\tilde{x}, \tilde{y}) \) of existing FIN-FACT examples are created using controlled edits:

- **Entity-level edits**: Swap company names or tickers with peers that have different financial outcomes.
- **Temporal edits**: Change dates or reporting periods to ones where the underlying financial condition changes.
- **Magnitude edits**: Perturb numeric values (e.g., profit, growth rates) to transform a true statement into a false one or vice versa.

Edits can be generated programmatically with heuristics or by prompting a strong teacher model (e.g., GPT-4o) to rewrite claims and contexts while specifying the desired new label, then verifying correctness via additional checks.

### 7.2 Training Counterfactual Robustness

The model is jointly trained on original and counterfactual examples, with two objectives:

1. Correctly classify the counterfactual label \( \tilde{y} \).
2. Generate explanations that highlight the edited components as the reason for label change (or invariance, in cases where the label is preserved).

A contrastive loss can be added that encourages the difference between original and counterfactual explanations to align with the underlying edits, encouraging explanations that articulate "what would need to change" for the claim to flip.

### 7.3 User-Facing Counterfactuals

At inference time, for high-stakes or ambiguous claims (e.g., with low confidence or near decision boundaries), the system can optionally generate a short counterfactual explanation such as:

- "If the reported net income had actually decreased year-on-year instead of increasing, this claim would be false, because the reasoning hinges on positive growth."

This adds an intuitive, forward-looking component to explanations without changing the core FMD task definition.

## 8. Integration with the Existing GPT-4o-mini-IT Pipeline

The proposed methodology is designed to be a drop-in extension of the current instruction-tuning pipeline rather than a complete architectural overhaul.[^1]

- The existing message structure (system prompt, user with updated claim and context, assistant with label and evidence) is retained.
- Additional signals (rationale spans, sufficiency/comprehensiveness variants, counterfactual examples) are incorporated during training via auxiliary heads and multi-view inputs, but inference can still operate with a single forward pass of GPT-4o-mini-IT.
- For deployment under FMD constraints, the system can serialize rationales into the explanation text (e.g., by quoting or marking key sentences) while using the internal span representation for faithfulness checks.

This makes the approach practical in settings where only a single text output channel is allowed, but richer internal explainability is desired.

## 9. Evaluation Plan for Explainability

Beyond the existing F1 and ROUGE metrics, CGSE enables a richer evaluation protocol:

1. **Rationale quality**
   - Token/sentence-level precision, recall, and F1 against aligned pseudo-gold rationales derived from the FIN-FACT evidence field.[^1]
   - Human evaluation on a subset to verify that highlighted spans are minimal yet sufficient to justify labels.

2. **Faithfulness metrics**
   - Quantitative sufficiency and comprehensiveness scores, measuring how much label confidence changes when rationales are removed or isolated.
   - Agreement between labels predicted from explanations alone and primary labels.

3. **Counterfactual robustness**
   - Accuracy on counterfactual examples and correlation between description of edits in explanations and actual edited elements.

4. **Human-centered evaluation**
   - Expert assessment by financial analysts of explanation usefulness for auditing decisions and understanding subtle financial nuances.

These evaluations complement ROUGE by directly measuring whether explanations are not only fluent and similar to reference text but also faithful, grounded, and informative for end users.

## 10. Summary

The CGSE methodology extends the existing GPT-4o-mini-IT financial misinformation detector with span-level rationales, faithfulness-enforcing objectives, structured explanation generation, and counterfactual reasoning while preserving compatibility with the FIN-FACT task setup. By turning human-written evidence texts into weak supervision for rationales and tying explanations more tightly to prediction behavior, the approach aims to produce explanations that are more interpretable, auditable, and trustworthy for regulators, financial institutions, and individual investors.[^1]

---

## References

1. [2025.finnlp-1.37-1.pdf](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/1152546/fb12e0f0-c9ea-48ea-a87e-bde841d0f965/2025.finnlp-1.37-1.pdf?AWSAccessKeyId=ASIA2F3EMEYEQGVSFCNB&Signature=DUOZcYigjNPrfSbHDUtlC9g0ri8%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEKD%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD9g2DmQ33JazvapZAqLLuUsbB9d%2B9jISbQLPhk15Ti%2BQIgFTIYQKlN3g%2F0fJI4kXoo%2F%2BsZOWy0z5QmXN7q3DHw%2FWUq8wQIaRABGgw2OTk3NTMzMDk3MDUiDDwhK1UfWv4fRbCD9yrQBG3Qwixh49rfTcb5BiMBlSdkqu4QHZgfspfufPMqGzIJ7fIHFA9nTMnyCyFfCQ70DysOzD1h6AxxHGGmJjKcE8ean1iG511KnBWFttmpKItgTlF8bCXfouM3AscFXs5BGWcIMv67ZvER3Lq1QfbbimfoZJRXpqOv5gnw5L546S8d5WSTd25i3y8DTUamCAXGmXulw5GXqLQX75NkF5mYdCMiYJ21kYNTM81R%2FgGwBk%2BYLfe6Rq4vrLrg54gbmFG03fz8AKBMJc9tXX0NWGP6JzHlLbaUiZF8g8qd2rn0R47dSDiTqCrLRpM42G11knUB0NB%2FA5Yhci0Ts%2Fl9%2FbFlX6b%2BjJbwU5mEeidSrx%2BKogOsfNJH9r7Rde%2B1fvfwRlfmV4Gd8P3Ni%2FzpLXxV6lPwfpYZF7aLl1qNmRZQlJf9u1gSmWV34tj%2F4tHnKXZz%2BAzZvQJ26elXyhxC2N03wX%2BWOoPe%2F3rxLDMzmkD6GSBk7yveVtfjXF6HhFN0JP28iehiAd8dttY1d%2FnVmVVkbobSzZtJ8ulj%2FmwvtHLIzwCejx9lZdsUN865G2GkBYYLX5pEts%2Br6dxGaheZTXu81DQ5VpYqGz0J%2Fw%2BLmlWk3hoJcCUXPhPJZbLNfRf3LQXMWjZNkVkSrgjQHWiST4da3YC6x21sxyRt9m2Je7ttIEJ%2B%2BiTq4Xkmaj9u6QW3q5zpR9KVfro4CJOhbqixLDlN2oMEN9pW3Y2nQh6MdDTr4nZMKsfd%2B4uA19h3xzW49r98LQJTtdln9%2F6HEJagcaxont4LO%2BowufiozwY6mAHvNVbEcL1CxGbNJZiIp7GmlK%2FZxPz2BUS4AhRLOnBvYKxPsu83IgEtU3aefhzKcEQYrdWl8TfOg1IN%2Bs44XSIBfCSkQx9QgKlOedjYM2Sh1eObh20mIo5dWxpeRIzsoaocKgHn5Gk7z6WNBgPwofLCpVIgVoqiAROCySekyVDdjhdcPU629slXzg5vak8kgklkmGiEPVTNLw%3D%3D&Expires=1776962060) - **page-1**
Deloitte (Drocks) at the Financial Misinformation Detection Challenge
Task: Enhancing Mis...

