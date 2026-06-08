# Ingestion Pipelines and Multi-Source Parsing Architectures

To build a robust, local WhatsApp analytics application, the system must process data from diverse, non-standard source formats across different platforms and localized regional settings. WhatsApp export files do not adhere to a single, unified structure. Device region settings, localized AM/PM designations, and system-level differences between iOS and Android result in widely varying plaintext schemas. The parsing engine must defensively ingest these logs, preserving chronological sequence and metadata integrity while avoiding data loss.

The system architecture supports two primary ingestion pathways: raw plaintext exports (TXT/ZIP) and direct extraction from local SQLite databases (msgstore.db and wa.db). Plaintext parsing is highly accessible to end-users but requires complex regular expressions to normalize timezone changes, Unicode artifacts, and multiline variations. In contrast, direct database extraction provides access to rich metadata—including conversation balance metrics, reaction data, and deleted message footprints—without relying on plaintext regex matching.

| Feature Dimension | Plaintext Export Pathway (TXT/ZIP) | Database Ingestion Pathway (SQLite) |
|---|---|---|
| Data Origin | Built-in WhatsApp "Export Chat" feature | Local database backups (msgstore.db, wa.db) |
| Extraction Schema | Plaintext lines or zipped folders | Normalized jid, chat_row_id, and LID mappings |
| Normalization Complexity | High; requires parsing custom date-time patterns | Low; queries structured tables using standard SQL |
| System Messages | Parsed from text lines (e.g., encryption notices) | Retained as distinct, categorized message types |
| Deleted Message Metadata | Excluded or marked as "This message was deleted" | Traceable via database schemas |
| Computational Overhead | High regex parsing and universal line parsing | Highly efficient SQL indexing and query execution |
| Privacy & PII Exposure | High risk if raw text is processed via external engines | Low risk; runs entirely locally on raw binary files |

The parsing engine must address several critical Unicode and formatting anomalies:

- **Invisible Unicode Characters:** iOS plaintext exports prepend a zero-width left-to-right mark (LRM), represented by the Unicode character `‎`, to date brackets and attachment markers. If this character is not programmatically stripped during preprocessing, regex patterns designed for standard text will fail on iOS logs.

- **Multiline Message Truncation:** Messages containing explicit newlines will break naive line-by-line parsers. To prevent this, the parser must implement lazy-matching algorithms and lookahead operations to verify if a line begins with a valid timestamp. If no timestamp is detected, the text is appended to the preceding message body.

- **Delimiter Ambiguity:** Displays names often contain colons, which can cause standard splitting functions to misclassify the sender field (e.g., splitting `User One: :` into `User One: :`). The parser must target only the first colon that marks the message boundary.

- **System Message Isolation:** WhatsApp inserts non-user events (such as security changes, call records, and group creations) directly into the timeline. These must be identified and isolated in the database to prevent them from skewing user-specific metrics.

For media-enriched exports, the system uses a specialized audio transcription pipeline to process voice notes. These recordings are saved as `.opus` files within an OGG container, named using distinct platform-specific schemas. Android uses a Push-to-Talk prefix (`PTT-YYYYMMDD-WAXXXX.opus`), while iOS formats the filename with an explicit audio indicator (`AUDIO-YYYY-MM-DD-HH-MM-SS.opus`).

Because local Whisper speech-to-text models require standard linear pulse-code modulation (PCM), the pipeline executes a conversion subprocess via FFmpeg. This transcodes the audio to 16 kHz mono WAV format before running inference:

$$\text{Opus Input Container} \xrightarrow{\text{FFmpeg Subprocess Execution}} \text{PCM WAV }(16\text{ kHz, Mono}) \xrightarrow{\text{Local Whisper Inference}} \text{Plaintext Transcript} \quad [2]$$

This pipeline ensures that audio-based interactions are transcribed locally and mapped directly to the corresponding timestamp in the conversation DataFrame.

---

# Cognitive, Emotional, and Coping Dynamics

## Ekman Emotion Classification and Emoji Synthesis

The emotion engine processes normalized text transcripts using a localized, distilled transformer architecture. The system utilizes `j-hartmann/emotion-english-distilroberta-base`, which classifies text into Ekman's six basic emotions (anger, disgust, fear, joy, sadness, surprise) along with a neutral baseline. This distilled model runs efficiently on local client CPUs, eliminating the latency and privacy risks of external APIs.

For setups with higher computational capacity, the system can use ModernBERT-Large. This architecture provides deeper contextual representations, helping identify implicit emotional cues in multi-turn conversations.

In digital communication, emojis carry significant emotional weight and alter the meaning of surrounding text. To capture this context, the system maps the transformer's emotion scores to an emoji sentiment scoring dictionary based on the Emoji Sentiment Ranking dataset. The text classification vector $T(m)$ and the emoji sentiment vector $S(e)$ are combined using a weighted hybrid formula to compute the composite emotional state $E_c(m)$ for each message $m$:

$$E_c(m) = w_t T(m) + w_e \sum_{e \in m} S(e) \quad [13, 17]$$

Here, $w_t$ and $w_e$ represent weighting parameters adjusted to balance textual context and explicit graphical symbols. This prevents misclassifications in highly expressive or non-verbal exchanges.

## Cognitive Coping via the DRIVE Framework

Standard sentiment classification maps text polarity to a simple positive-to-negative scale, which often fails to capture the nuances of psychological coping. To address this, the system incorporates the Demands-Resources-Individual Effects (DRIVE) model. This framework categorizes conversational text into cognitive demands, coping mechanisms, and available personal resources.

Coping mechanisms are classified into positive and negative strategies:

- **Positive Coping:** Characterized by self-care, seeking external help, positive reframing, mindfulness, and the use of adaptive humor or sarcasm.
- **Negative Coping:** Characterized by hopeless projection, negative perceptions, conspiracy theories, and wishful or avoidant thinking.

Analyzing coping strategies provides a much more accurate assessment of a user's psychological state than simple sentiment mapping. For example, a user sharing a dark joke or venting about a stressful event may generate a negative sentiment score (via standard polarity tools like VADER or TextBlob). However, within the DRIVE model, this behavior is correctly identified as an active, positive coping strategy (adaptive humor).

| Coping Classification | Specific Sub-theme | Behavioral & Conversational Indicators |
|---|---|---|
| Positive Coping (70%) | Self-Care / Active Framing | Prioritizing rest, setting physical boundaries, and establishing positive routines. |
| | Seeking Professional Help | Discussing therapy appointments, scheduling medical consults, or seeking guidance. |
| | Prayer and Meditation | Expressions of spiritual reflection, mindfulness exercises, or deep breathing routines. |
| | Adaptive Humor / Sarcasm | Using self-deprecating humor or dry wit to reduce the tension of stressful events. |
| Negative Coping (30%) | Conspiracy / Paranoia | Blaming external organizations, projecting hidden motives, or displaying severe distrust. |
| | Hopeless / Passive Thinking | Expressing resignation, assuming outcomes cannot be changed, or displaying helplessness. |
| | Avoidance / Wishful Thinking | Disengaging from problems, withdrawing from interactions, or relying on escapism. |

To handle the highly imbalanced nature of real-world text datasets, the classification models are optimized using Focal Loss during local fine-tuning. Focal Loss addresses class imbalance by dynamically down-weighting the loss contribution of easy-to-classify examples, forcing the model to focus on rare but critical emotional states:

$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad [28]$$

Here, $p_t$ represents the model's estimated probability for the correct class, $\alpha_t$ balances class distributions, and $\gamma$ is a focusing parameter that adjusts the rate at which easy examples are down-weighted.

## Local LLM Coping Strategy Orchestration via Gemma

While sequence classification models run efficiently, they struggle with complex, highly localized context or implicit conversational sarcasm. To resolve these limitations, the application features an optional orchestration tier leveraging a locally installed Gemma LLM executing via Ollama.

Running locally on the user's desktop client, Gemma acts as a high-fidelity reasoning agent. The system passes target messages flagged with complex emotional profiles directly to Gemma, bypassing simple regex or static pipelines:

$$\text{Flagged Chat Message} \xrightarrow{\text{Ollama Local Port Client}} \text{Gemma Context Model} \xrightarrow{\text{Structured JSON Output}} \text{DRIVE Coping Assessment}$$

Using local LLM orchestration allows the system to determine why a particular coping strategy is active, documenting structural transitions and logging behavioral patterns without violating strict personal data privacy boundaries.

---

# Topic Identification, Latent Dirichlet Allocation, and Entity Mapping

## Topic Discovery via LDA and Dynamic Topic Modeling

To extract the primary discussion themes from the chat corpus, the system implements a localized Latent Dirichlet Allocation (LDA) engine. This probabilistic model discovers hidden thematic structures by treating conversations as a mixture of topics, where each topic is defined by a distinct probability distribution over words.

To track how discussion topics evolve over time, the system uses a Python-based Dynamic Topic Modeling (DTM) algorithm. This approach divides the conversation timeline into discrete intervals, allowing the model to trace shifts in thematic focus. In clinical and supportive contexts, this helps track transitions between active crisis states and long-term recovery:

```
Time Intervals → DTM Engine
                 │
                 ▼
    Discrete Time Slices (T1, T2, T3...)
                 │
                 ▼
    Per-Slice Topic Distributions
                 │
                 ├─► Topic 1 (Medication & Clinical Care)
                 ├─► Topic 2 (Interpersonal Support)
                 ├─► Topic 3 (Coping & Treatment Focus)
                 ▼
   ──► Pathology-focused care vs. supportive care
```

By tracing these dynamic topic paths, the system can distinguish between pathology-focused trajectories (such as conversations centered on symptoms or medication) and supportive care environments (conversations focused on relationship-building and positive coping).

## Aspect-Based Sentiment Analysis (ABSA)

To map conversational sentiment to specific people, entities, or topics, the system uses Aspect-Based Sentiment Analysis (ABSA). Unlike standard sentiment analysis, which assigns a single polarity score to an entire message, ABSA isolates specific terms within a sentence and evaluates the sentiment toward each target. For example, in the message "The screen on this phone is gorgeous, but the battery life is terrible," the system extracts two distinct aspect-sentiment pairs:

$$\text{screen} \longrightarrow \text{Positive} \quad | \quad \text{battery life} \longrightarrow \text{Negative} \quad [30]$$

To analyze complex sentences where standard token classifiers might drop nuanced links, the local Gemma model can be invoked via Ollama using structured generation options (e.g., forced JSON schemas). Gemma processes multi-sentence context, extracts named individuals and topics (e.g., "Mother," "Boss," "Project Alpha"), and maps targeted emotional valences to each entity with deep reasoning capabilities.

| ABSA Engine Model | Underlying Architecture | Operational Mechanics & Token Tagging | Character Offsets | Target Device Scope |
|---|---|---|---|---|
| SetFitABSA | Few-shot Sentence Transformer | Extracts candidate aspect spans using spaCy, filters targets via SetFit, and classifies polarities. | Spans identified via spaCy token intervals. | Optimized for local CPUs and low-resource environments. |
| DeBERTa-v3 ABSA | Joint Token-Classification Head | Jointly extracts aspects and sentiments using an IOB-with-sentiment tag schema (e.g., B-ASP-Positive). | Returns exact start and end character offsets. | Optimized for local GPUs or high-performance desktop environments. |
| Gemma LLM (Ollama) | Generative Decoder Model (7B/9B) | Parses conversational segments, identifies abstract entities, and maps emotional associations via natural language parsing. | Structured JSON schema boundaries (key-value mapping). | Local high-end CPU / Apple Silicon / Dedicated local GPU. |

To maintain high classification accuracy, the text must undergo targeted preprocessing. Standard NLP cleaning steps—such as lowercasing or removing punctuation—must be applied selectively so they do not inadvertently alter the meaning of the text:

- **Negation Processing:** Words like "not", "no", and "never" must be preserved, and common contractions must be explicitly expanded (e.g., "weren't" → "were not"). This step is critical for preventing the model from misattributing sentiment polarities.
- **Intensifier Modeling:** Words that amplify expression (such as "very", "extremely", or "completely") must be retained, as they provide essential context for sentiment classification models.
- **Emoji-to-Token Mapping:** Emojis are converted into explicit textual representations to preserve their sentiment indicators (e.g., converting 🤬 into "angry face token").

---

# Quantitative Crisis Identification and Warning Sign Parsing

## Machine Learning Classification of Distress Signals

To monitor conversational health effectively, the system implements a localized, two-tier crisis detection engine. Tier 1 uses a fast, regex-based keyword matching layer that identifies explicit risk signals. Tier 2 uses a machine learning classifier to detect implicit expressions of severe distress, worthlessness, and helplessness.

Data preprocessing for the machine learning model must be fast and reproducible. Raw strings are lowercased, and usernames, URLs, and numeric characters are removed. To preserve syntactic intent, repeated characters (e.g., "soooo" → "so") are normalized to single letters. Finally, the text is tokenized, common stopwords are removed, and the remaining tokens are stemmed using a Porter Stemmer.

To convert the cleaned text into numerical features, the system computes $TF\text{-}IDF$ weights for the extracted words:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \log\left(\frac{|D|}{1 + |\{d \in D : t \in d\}|}\right) \quad [35]$$

The resulting feature matrix is processed by a local Random Forest Classifier. This classifier is chosen for its efficiency, low memory footprint, and ability to process high-dimensional sparse text vectors without the need for high-end local hardware. The model achieves strong predictive performance, balancing precision and recall to minimize false positives:

$$\text{Crisis Predictor }\mathcal{M} \longrightarrow \begin{cases} \text{Accuracy: } 85\% \\ \text{Precision: } 88\% \\ \text{Recall: } 83\% \end{cases} \quad [26, 35]$$

This ensures reliable crisis detection while avoiding unnecessary alert fatigue.

## Warning Sign Taxonomy and Intervention Protocols

The system categorizes crisis warning signs into four distinct psychological dimensions. The warning sign matrix below outlines these categories, along with high-frequency risk keywords and their associated warning levels:

| Diagnostic Dimension | Warning Signs & Indicators | High-Frequency Match Tokens | Emergency Level Assignment |
|---|---|---|---|
| Cognitive | Expressions of unworthiness, failure, hopelessness, or feeling like an intolerable burden. | "hopeless", "worthless", "burden", "no point", "unworthy", "no reason to live". | Level 2: Elevated Risk (Initiate active tracking and flag emotional shift logs). |
| Physiological | Mentions of physical distress, severe sleep disturbances (insomnia/hypersomnia), chronic exhaustion, or physical panic symptoms. | "can't sleep", "exhausted", "headache", "tremble", "shiver", "dyspnea", "heart racing". | Level 1: Moderate Risk (Classify under somatic stressors within the DRIVE framework). |
| Behavioral | Discussion of self-harm, acquiring weapons or materials, making end-of-life preparations, or sudden social withdrawal. | "social withdrawal", "want to be alone", "goodbye", "make a plan", "cutting", "will". | Level 3: Critical Risk (Trigger local emergency resource routing). |
| Emotional | Expressions of overwhelming emotional pain, intense crying, rage, severe panic, or sudden, unexplained mood shifts. | "crying", "rage", "humiliation", "panic", "empty", "angry", "uncontrolled anger". | Level 2: Elevated Risk (Correlate with negative coping metrics). |

To analyze the severity of flagged warning signs, the system uses rank-order centrality metrics. These metrics measure how closely warning words correlate with active crisis states in conversational data:

$$\text{Low Mood} \longrightarrow C_D = 21, \text{ closeness} = 0.717 \quad | \quad \text{Social Withdrawal} \longrightarrow C_D = 15, \text{ closeness} = 0.635 \quad [27]$$

This rank-order scoring allows the model to prioritize warning signs that correlate most strongly with acute psychological distress.

## Tier 3 Validation and Context Verification via Gemma

To avoid catastrophic misclassifications—such as flag-matching standard idioms or sarcastic phrases (e.g., "This homework is killing me" vs. "I want to die")—the engine deploys Gemma as a Tier 3 context validator.

When a Level 2 or Level 3 crisis trigger is flagged by simple pattern matching or random forest architectures, the system routes the preceding conversation window (up to five multi-turn messages) to the local Gemma instance. Gemma is prompted to evaluate the context and determine if the threat is literal, metaphorical, or conversational noise. This hybrid classification design minimizes false-positive interruptions, protecting diagnostic validity.

For confirmed high-risk states (validated Level 3), the application initiates a local safety response. This response bypasses external networks to surface verified regional support services and emergency phone numbers, ensuring a reliable, private first-line safety mechanism.

---

# Interactive Network Topology and Visualizations

## Graph Construction and Centrality Modeling

To represent conversational dynamics visually, the system constructs a directional, weighted graph model using NetworkX. In this network, nodes represent individuals or entities mentioned in the chat, and edges represent communication pathways and sentiment attributes.

The mathematical characteristics of the graph are defined as follows:

Let $G = (V, E)$ be a directed graph where $V$ is the set of all unique speakers and extracted entities:

$$V = \{v_1, v_2, \dots, v_n\} \quad [39]$$

An edge $e_{ij} = (v_i, v_j)$ exists if $v_i$ sends a message to or references $v_j$. Each edge is assigned several key properties:

- **Weight ($W_{ij}$):** Represents the interaction frequency or total message volume between the nodes.
- **Sentiment Value ($S_{ij}$):** The average polarity score of the interactions, mapped on a continuous scale: $S_{ij} \in [-1.0, 1.0] \quad [24, 40]$
- **Temporal Response Latency ($T_{ij}$):** The median response time between the sender's message and the receiver's reply, capturing the behavioral balance of the exchange.

Node size is determined by its normalized Degree Centrality $C_D(v)$, which measures the node's relative importance within the network:

$$C_D(v) = \frac{\text{deg}(v)}{N - 1} \quad [39, 40]$$

To identify distinct conversational groups and sub-clusters, the system applies community detection using the Louvain algorithm (`community.best_partition`). This partition partitions the network by optimizing its modularity score.

## Interactive Plotly Rendering and Topology Encoding

Because Plotly does not have native support for complex graph layouts, the system uses NetworkX to pre-calculate node coordinates before rendering. It calculates a force-directed layout using a spring layout algorithm:

$$\text{pos} = \text{nx.spring\_layout}(G, k=1/\sqrt{N}, \text{seed}=42) \quad [39, 42, 44]$$

This coordinate layout is then converted into interactive Plotly traces:

- **Edges:** Rendered as separate `go.Scatter` trace markers with a `mode='lines'` configuration. The visual thickness of each line represents the connection weight $W_{ij}$.
- **Nodes:** Rendered as a `go.Scatter` trace with `mode='markers'`. The node color maps directly to the average sentiment score (using a red-to-green gradient), and the marker size scales with the node's degree centrality.
- **Interactions:** Nodes are populated with rich hover tooltips containing the entity name, connection count, and average sentiment score, allowing users to explore the network dynamically.

---

# Technical Specifications and Claude Code Integration Specifications

This section provides the technical implementation blueprints designed to guide development with Claude Code. It includes code blocks for the core parser, the emotional feature aggregator, the Gemma integration pipeline, and the network visualizer.

## Programmatic Regex Generation and Multi-Format Parser

This module generates platform-specific regex patterns from a simplified header syntax, parsing raw transcripts into normalized pandas DataFrames.

```python
import re
import os
import unicodedata
from datetime import datetime
import pandas as pd

class ProgrammaticWhatsAppParser:
    """
    Parser for converting raw WhatsApp exports into normalized pandas DataFrames.
    Includes support for custom header formatting and multi-line parsing.
    """
    def __init__(self, filepath: str, hformat: str = None, encoding: str = "utf-8"):
        self.filepath = filepath
        self.encoding = encoding
        self.hformat = hformat or "%d/%m/%y, %H:%M - %name:"  # Default fallback format [47]
        
        # Mapping table translating simplified format tokens to regex patterns [8]
        self.regex_map = {
            "%Y": r"(?P<year>\d{2,4})",
            "%y": r"(?P<year>\d{2,4})",
            "%m": r"(?P<month>\d{1,2})",
            "%d": r"(?P<day>\d{1,2})",
            "%H": r"(?P<hour>\d{1,2})",
            "%I": r"(?P<hour>\d{1,2})",
            "%M": r"(?P<minutes>\d{2})",
            "%S": r"(?P<seconds>\d{2})",
            "%P": r"(?P<ampm>[AaPp]\.?\s?[Mm]\.?)",
            "%p": r"(?P<ampm>[AaPp]\.?\s?[Mm]\.?)",
            "%name": r"(?P<username>[^:]*)"
        }

    def generate_regex_patterns(self) -> tuple:
        """
        Translates a simplified header format string into a compilable regex.
        Matches whatstk's generation syntax.
        """
        pattern = self.hformat
        # Extract and replace format tokens [8]
        tokens = re.findall(r"%\w*", pattern)
        for t in tokens:
            if t in self.regex_map:
                pattern = pattern.replace(t, self.regex_map[t])
        
        # Ensure regex ends with space normalization
        pattern = pattern + " "
        # Generate target prefix pattern by splitting at username [8]
        prefix_pattern = pattern.split("(?P<username>[^:]*)")
        return pattern, prefix_pattern

    def parse_transcript(self) -> pd.DataFrame:
        """
        Parses raw text transcripts, resolving Unicode artifacts and multi-line message wraps.
        """
        full_pattern, prefix_pattern = self.generate_regex_patterns()
        msg_regex = re.compile(f"^{full_pattern}")
        prefix_regex = re.compile(f"^{prefix_pattern}")

        parsed_records = []
        current_record = None

        with open(self.filepath, "r", encoding=self.encoding, errors="replace") as file:
            for raw_line in file:
                # Strip Unicode artifacts like iOS left-to-right marks
                cleaned_line = raw_line.replace("‎", "").replace("‏", "")
                
                # Check for a new message boundary
                match = msg_regex.match(cleaned_line)
                if match:
                    if current_record:
                        parsed_records.append(current_record)
                    
                    data = match.groupdict()
                    # Resolve date values defensively [2, 5]
                    year = int(data["year"]) + 2000 if len(data["year"]) == 2 else int(data["year"])
                    dt = datetime(
                        year=year,
                        month=int(data["month"]),
                        day=int(data["day"]),
                        hour=int(data["hour"]),
                        minute=int(data["minutes"]),
                        second=int(data.get("seconds", 0))
                    )
                    
                    # Extract raw message text [4, 12]
                    span_end = match.end()
                    raw_text = cleaned_line[span_end:].strip()
                    
                    # Isolate system messages from standard user chats
                    sender = data.get("username", "System Message").strip()
                    is_system = sender == "System Message" or "joined" in raw_text or "changed" in raw_text

                    current_record = {
                        "date": dt,
                        "username": sender,
                        "message": raw_text,
                        "message_type": "system" if is_system else "user"  # [8]
                    }
                else:
                    # Append multi-line content to the active message [11, 14]
                    if current_record:
                        current_record["message"] += " " + cleaned_line.strip()
                    else:
                        continue

            if current_record:
                parsed_records.append(current_record)

        return pd.DataFrame(parsed_records)
```

## Local Affect and Emotional Feature Aggregator

This module integrates the local transformer pipeline with emoji-based scoring to calculate emotional valence.

```python
import emoji
from transformers import pipeline

class LocalAffectAggregator:
    """
    Computes Ekman emotional states and maps emoji sentiment patterns locally.
    """
    def __init__(self):
        # Local pipeline instance utilizing DistilRoBERTa
        self.classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        # Weight mapping derived from Emoji Sentiment rankings
        self.emoji_lexicon = {
            "😭": -0.85, "🤬": -0.90, "🤢": -0.70, "😨": -0.65,
            "🥰": 0.90,  "😍": 0.85,  "😀": 0.75,  "👍": 0.70,
            "😐": 0.00,  "😲": 0.25
        }

    def compute_message_affect(self, text: str) -> dict:
        # Preprocess text and isolate emoji tokens
        extracted_emojis = [char for char in text if emoji.is_emoji(char)]
        clean_text = emoji.replace_emoji(text, replace="").strip()

        # Compute text-based emotions if message body contains content
        scores = {e: 0.0 for e in ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]}
        if clean_text:
            try:
                preds = self.classifier(clean_text)
                for p in preds:
                    scores[p["label"]] = p["score"]
            except Exception:
                scores["neutral"] = 1.0
        else:
            scores["neutral"] = 1.0

        # Calculate emoji sentiment score
        emoji_valences = [self.emoji_lexicon.get(em, 0.0) for em in extracted_emojis]
        avg_emoji_sentiment = sum(emoji_valences) / len(emoji_valences) if emoji_valences else 0.0

        # Compute overall emotional valence
        composite_valence = 0.7 * (scores["joy"] - scores["sadness"]) + 0.3 * avg_emoji_sentiment

        return {
            "text_scores": scores,
            "emoji_count": len(extracted_emojis),
            "emoji_sentiment": avg_emoji_sentiment,
            "composite_valence": composite_valence
        }
```

## Local Gemma Inference and Cognitive Reasoning Module

This module utilizes the `ollama` Python library to communicate directly with your local Gemma instance for structured ABSA, DRIVE taxonomy mapping, and Tier 3 validation.

```python
import json
import ollama

class LocalGemmaInferencePipeline:
    """
    Orchestrates complex semantic reasoning over conversational segments using
    a locally running Gemma LLM via Ollama. Handles zero-shot structured tasks.
    """
    def __init__(self, model_name: str = "gemma3"):
        self.model_name = model_name

    def analyze_message_context(self, context_messages: list) -> dict:
        """
        Processes a sliding window of conversational context to extract structured aspects,
        coping classifications, and check implicit risk warning markers.
        """
        # Format the sliding history into a clear prompt block
        formatted_history = "\n".join([f"[{m['sender']}]: {m['text']}" for m in context_messages])
        
        prompt = f"""
        Perform a thorough structural text analysis of the following local chat context. 
        Evaluate specific entities/people mentioned, the DRIVE coping mechanisms shown by participants,
        and verify if any perceived risk warning elements are actively present.

        Analyze this chat context:
        ---
        {formatted_history}
        ---

        You must respond STRICTLY with a valid, clean JSON object matching this structural schema. 
        Do not prepend any markdown text, instructions, or wrappers other than the raw JSON itself:
        {{
            "aspect_sentiments": [
                {{"target_entity": "Name of person/topic", "valence": -1.0 to 1.0, "reasoning": "Context why"}}
            ],
            "coping_analysis": {{
                "dominant_strategy": "Positive|Negative|None",
                "sub_theme": "Self-Care|Seeking Help|Prayer|Adaptive Humor|Conspiracy|Hopeless|Avoidance|None",
                "explanation": "Brief breakdown of the psychological coping state"
            }},
            "crisis_validator": {{
                "is_literal_risk": false,
                "confidence_score": 0.0 to 1.0,
                "contextual_notes": "Sarcasm vs literal verification"
            }}
        }}
        """

        try:
            # Query the local Ollama daemon client with target model and system templates
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,  # Keep outputs highly structured and deterministic
                    "num_predict": 512
                }
            )

            # Extract raw textual content from the completion payload
            raw_content = response.message.content.strip()
            
            # Clean possible markdown wrap-around errors
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:-3].strip()
            elif raw_content.startswith("```"):
                raw_content = raw_content[3:-3].strip()

            return json.loads(raw_content)

        except Exception as e:
            # Safe programmatic fallback if local model throws error or is offline
            return {
                "error": f"Local Gemma processing failure: {str(e)}",
                "aspect_sentiments": [],
                "coping_analysis": {
                    "dominant_strategy": "None",
                    "sub_theme": "None",
                    "explanation": "Model execution error fallback."
                },
                "crisis_validator": {
                    "is_literal_risk": False,
                    "confidence_score": 0.0,
                    "contextual_notes": "Execution error, fallback to standard regex/random forest."
                }
            }
```

## Dynamic Network and Interactive Topology Generator

This module constructs an entity-sentiment graph using NetworkX and renders it as an interactive Plotly visualization.

```python
import networkx as nx
import plotly.graph_objects as go

class InteractiveTopologyGenerator:
    """
    Constructs directional, weighted interaction graphs from extracted 
    entity-sentiment records and renders them as interactive Plotly visualizations.
    """
    def __init__(self, target_records: list):
        self.records = target_records
        self.G = nx.Graph()

    def construct_interaction_graph(self):
        """
        Builds a NetworkX graph from target records.
        """
        for r in self.records:
            source = r["sender"]
            target = r["entity"]
            valence = r["valence"]  # Continuous scale from -1.0 to 1.0 [24]

            if not self.G.has_edge(source, target):
                self.G.add_edge(source, target, weight=1, valences=[valence])
            else:
                self.G[source][target]["weight"] += 1
                self.G[source][target]["valences"].append(valence)

    def generate_plotly_figure(self) -> go.Figure:
        """
        Calculates node coordinates and renders the graph trace.
        """
        self.construct_interaction_graph()
        
        # Calculate force-directed node coordinates using spring layout
        pos = nx.spring_layout(self.G, k=1.0, seed=42)

        # Generate edge trace [45]
        edge_x = []
        edge_y = []
        for u, v in self.G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.8, color="#a6a6a6"),
            hoverinfo="none",
            mode="lines"
        )

        # Generate node trace [45]
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        # Calculate centralities to determine node sizes
        centrality_map = nx.degree_centrality(self.G)

        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Map marker size to normalized degree centrality [39, 41]
            c_val = centrality_map[node]
            node_size.append(15 + (c_val * 45))

            # Calculate average sentiment toward the entity
            all_valences = []
            for neighbor in self.G.neighbors(node):
                all_valences.extend(self.G[node][neighbor]["valences"])
            avg_valence = sum(all_valences) / len(all_valences) if all_valences else 0.0
            node_color.append(avg_valence)

            node_text.append(
                f"Entity Target: {node}<br>"
                f"Degree Centrality: {c_val:.3f}<br>"
                f"Computed Valence: {avg_valence:.2f}"
            )

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale="RdYlGn",  # Red-yellow-green colormap for sentiment polarity
                color=node_color,
                size=node_size,
                colorbar=dict(
                    title="Average Valence",
                    thickness=15,
                    titleside="right"
                ),
                line_width=1.5
            )
        )

        # Compile traces into final Figure [45]
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Conversational Entity-Sentiment Topology Map",
                titlefont_size=16,
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        return fig
```

---

# Local Development Guidelines for Claude Code

To guide development with Claude Code, the system should follow these structured implementation instructions:

- **Strict Local Environments:** Ensure all model dependencies—such as `transformers`, `spacy`, `setfit`, and the local `ollama` client utility wrapper—are configured for offline-only execution. No external cloud APIs are allowed.

- **Deterministic Test Suite:** Script reproducible test cases with dummy records to verify timezone normalization, Unicode character stripping (specifically LRM `‎`), and multiline wrap-around.

- **Local Model Optimization:** When loading NLP classifiers locally, configure standard quantization using ONNX runtimes. This optimizes memory usage and execution speed for low-resource environments.

- **Pipeline Guardrails:** Implement local guardrails using tools like `toxic-bert` and `Presidio`. This filters toxic inputs and ensures sensitive personally identifiable information (PII) is redacted locally before downstream processing.

---

# Conclusion and Strategic Architecture

Implementing a local WhatsApp behavioral analytics engine requires a careful balance between security, computational efficiency, and analytical depth. By utilizing a programmatic regex parser, the system can defensively ingest diverse chat archives while preserving chronological integrity. Combining Ekman emotion classification, the DRIVE coping framework, and aspect-based sentiment analysis enables deep psychological modeling.

By introducing your local Gemma LLM via Ollama, you enhance this architecture, allowing for robust multi-turn context checking and structured JSON generation without compromising user confidentiality. Finally, representing these interactions through NetworkX and interactive Plotly graphs turns unstructured conversation data into intuitive, actionable visualizations. This technical blueprint provides a secure, robust foundation for local behavioral text analytics.
