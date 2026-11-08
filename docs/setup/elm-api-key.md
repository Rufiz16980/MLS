# ELM API key (each person requests their own)

1. Open https://elm.edina.ac.uk/elm and sign in with your university account.
2. Request an API key and describe the intended use. Example (from the course slides):

> I am taking INFR11269 Machine Learning Systems at the University of Edinburgh. I will use my ELM API key for the course's retrieval-augmented generation (RAG) coursework and related systems experiments.
> I intend to use the OpenAI-compatible Chat Completions endpoint (/chat/completions) through https://elm.edina.ac.uk/api/v1. Requests will support two activities: generating answers from retrieved coursework documents, and evaluating candidate answers for correctness, grounding and completeness.
> The workflow starts with course-provided benchmark audio. Speech recognition runs locally; ELM requests contain the resulting text transcript, selected documents from the supplied course corpus and, for evaluation, candidate answers. I do not intend to upload raw audio, confidential university information, credentials or personal student records. Public source text may still contain personal references or sensitive topics, so I will review the data I submit and follow the course's data-use guidance.
> I will keep my key in a local environment variable or an excluded .env file, and will not include it in shared code, repositories, reports or prompts.
> I will limit requests to coursework experiments, use caching where appropriate, monitor usage and avoid unnecessary repeated calls. I will treat generated answers and automated evaluation scores as fallible, inspect representative outputs and acknowledge my use of AI in accordance with the course's academic-integrity requirements.

3. Put the key in `rag/.env` as `ELM_API_KEY=...`. Never commit it, paste it into chats, or share it with anyone or any AI tool.
4. Use the locally hosted free models (the code default). Switching models alone is not a contribution.
5. Limits: one key per person, valid one year, revoked after 2 months of inactivity, $500/year cap per key. Cache generations and judgements to save calls.

More: https://elm.edina.ac.uk/documentation/docs/13-keys/
