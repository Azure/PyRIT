API Reference
=============

:py:mod:`pyrit.analytics`
=========================

.. automodule:: pyrit.analytics
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    ConversationAnalytics

:py:mod:`pyrit.auth`
====================

.. automodule:: pyrit.auth
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    Authenticator
    AzureAuth
    AzureStorageAuth

:py:mod:`pyrit.auxiliary_attacks`
=================================

.. automodule:: pyrit.auxiliary_attacks
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/



:py:mod:`pyrit.chat_message_normalizer`
=======================================

.. automodule:: pyrit.chat_message_normalizer
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    ChatMessageNormalizer
    ChatMessageNop
    GenericSystemSquash
    ChatMessageNormalizerChatML
    ChatMessageNormalizerTokenizerTemplate

:py:mod:`pyrit.common`
======================

.. automodule:: pyrit.common
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/



:py:mod:`pyrit.datasets`
========================

.. automodule:: pyrit.datasets
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    fetch_decoding_trust_stereotypes_examples
    fetch_examples
    fetch_harmbench_examples
    fetch_many_shot_jailbreaking_examples
    fetch_seclists_bias_testing_examples
    fetch_xstest_examples
    fetch_pku_safe_rlhf_dataset
    fetch_adv_bench_dataset
    fetch_wmdp_dataset
    fetch_forbidden_questions_df
    fetch_llm_latent_adversarial_training_harmful_dataset
    fetch_tdc23_redteaming_dataset

:py:mod:`pyrit.embedding`
=========================

.. automodule:: pyrit.embedding
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AzureTextEmbedding
    OpenAiTextEmbedding

:py:mod:`pyrit.exceptions`
==========================

.. automodule:: pyrit.exceptions
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    BadRequestException
    EmptyResponseException
    handle_bad_request_exception
    InvalidJsonException
    MissingPromptPlaceholderException
    PyritException
    pyrit_json_retry
    pyrit_target_retry
    pyrit_placeholder_retry
    RateLimitException
    remove_markdown_json

:py:mod:`pyrit.memory`
======================

.. automodule:: pyrit.memory
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AzureSQLMemory
    CentralMemory
    DuckDBMemory
    EmbeddingDataEntry
    MemoryInterface
    MemoryEmbedding
    MemoryExporter
    PromptMemoryEntry

:py:mod:`pyrit.models`
======================

.. automodule:: pyrit.models
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    ALLOWED_CHAT_MESSAGE_ROLES
    AudioPathDataTypeSerializer
    AzureBlobStorageIO
    ChatMessage
    ChatMessagesDataset
    ChatMessageRole
    ChatMessageListDictContent
    construct_response_from_request
    DataTypeSerializer
    data_serializer_factory
    DiskStorageIO
    EmbeddingData
    EmbeddingResponse
    EmbeddingSupport
    EmbeddingUsageInformation
    ErrorDataTypeSerializer
    group_conversation_request_pieces_by_sequence
    Identifier
    ImagePathDataTypeSerializer
    PromptRequestPiece
    PromptResponse
    PromptResponseError
    PromptDataType
    PromptRequestResponse
    QuestionAnsweringDataset
    QuestionAnsweringEntry
    QuestionChoice
    Score
    ScoreType
    SeedPrompt
    SeedPromptDataset
    SeedPromptGroup
    StorageIO
    TextDataTypeSerializer
    UnvalidatedScore

:py:mod:`pyrit.orchestrator`
============================

.. automodule:: pyrit.orchestrator
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    CrescendoOrchestrator
    FlipAttackOrchestrator
    FuzzerOrchestrator
    MultiTurnAttackResult
    MultiTurnOrchestrator
    Orchestrator
    PAIROrchestrator
    PromptSendingOrchestrator
    RedTeamingOrchestrator
    ScoringOrchestrator
    SkeletonKeyOrchestrator
    TreeOfAttacksWithPruningOrchestrator
    XPIAManualProcessingOrchestrator
    XPIAOrchestrator
    XPIATestOrchestrator

:py:mod:`pyrit.prompt_converter`
================================

.. automodule:: pyrit.prompt_converter
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AddImageTextConverter
    AddTextImageConverter
    AsciiArtConverter
    AtbashConverter
    AudioFrequencyConverter
    AzureSpeechAudioToTextConverter
    AzureSpeechTextToAudioConverter
    Base64Converter
    CaesarConverter
    CharacterSpaceConverter
    CodeChameleonConverter
    ConverterResult
    EmojiConverter
    FlipConverter
    FuzzerCrossOverConverter
    FuzzerExpandConverter
    FuzzerRephraseConverter
    FuzzerShortenConverter
    FuzzerSimilarConverter
    HumanInTheLoopConverter
    LeetspeakConverter
    LLMGenericTextConverter
    MaliciousQuestionGeneratorConverter
    MathPromptConverter
    MorseConverter
    NoiseConverter
    PersuasionConverter
    PromptConverter
    QRCodeConverter
    RandomCapitalLettersConverter
    RepeatTokenConverter
    ROT13Converter
    SearchReplaceConverter
    StringJoinConverter
    SuffixAppendConverter
    TenseConverter
    ToneConverter
    TranslationConverter
    UnicodeConfusableConverter
    UnicodeSubstitutionConverter
    UrlConverter
    VariationConverter

.. automodule:: pyrit.prompt_converter.fuzzer_converter
    :no-members:
    :no-inherited-members:
    :no-index:

:py:mod:`pyrit.prompt_normalizer`
=================================

.. automodule:: pyrit.prompt_normalizer
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    PromptNormalizer
    PromptResponseConverterConfiguration
    NormalizerRequestPiece
    NormalizerRequest

:py:mod:`pyrit.prompt_target`
=============================

.. automodule:: pyrit.prompt_target
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AzureBlobStorageTarget
    AzureMLChatTarget
    CrucibleTarget
    GandalfLevel
    GandalfTarget
    HTTPTarget
    HuggingFaceChatTarget
    HuggingFaceEndpointTarget
    limit_requests_per_minute
    OllamaChatTarget
    OpenAICompletionTarget
    OpenAIDALLETarget
    OpenAIChatTarget
    OpenAITTSTarget
    OpenAITarget
    OllamaChatTarget
    PromptChatTarget
    PromptShieldTarget
    PromptTarget
    TextTarget

:py:mod:`pyrit.score`
=====================

.. automodule:: pyrit.score
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AzureContentFilterScorer
    ContentClassifierPaths
    FloatScaleThresholdScorer
    GandalfScorer
    HumanInTheLoopScorer
    LikertScalePaths
    MarkdownInjectionScorer
    PromptShieldScorer
    Scorer
    SelfAskCategoryScorer
    SelfAskLikertScorer
    SelfAskRefusalScorer
    SelfAskScaleScorer
    SelfAskTrueFalseScorer
    SubStringScorer
    TrueFalseInverterScorer
    TrueFalseQuestion
    TrueFalseQuestionPaths
