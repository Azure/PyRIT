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


:py:mod:`pyrit.cli`
=======================================

.. automodule:: pyrit.cli
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/


:py:mod:`pyrit.common`
======================

.. automodule:: pyrit.common
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    combine_dict
    combine_list
    convert_local_image_to_data_url
    deprecation_message
    display_image_response
    download_chunk
    download_file
    download_files
    download_specific_files
    get_available_files
    get_httpx_client
    get_kwarg_param
    get_non_required_value
    get_random_indices
    get_required_value
    initialize_pyrit
    is_in_ipython_session
    make_request_and_raise_if_error_async
    print_chat_messages_with_color
    Singleton
    warn_if_set
    YamlLoadable

:py:mod:`pyrit.datasets`
========================

.. automodule:: pyrit.datasets
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    fetch_adv_bench_dataset
    fetch_aya_redteaming_dataset
    fetch_babelscape_alert_dataset
    fetch_ccp_sensitive_prompts_dataset
    fetch_darkbench_dataset
    fetch_decoding_trust_stereotypes_dataset
    fetch_equitymedqa_dataset_unique_values
    fetch_examples
    fetch_forbidden_questions_dataset
    fetch_harmbench_dataset
    fetch_librAI_do_not_answer_dataset
    fetch_llm_latent_adversarial_training_harmful_dataset
    fetch_jbb_behaviors_by_harm_category
    fetch_jbb_behaviors_by_jbb_category
    fetch_jbb_behaviors_dataset
    fetch_many_shot_jailbreaking_dataset
    fetch_medsafetybench_dataset
    fetch_mlcommons_ailuminate_demo_dataset
    fetch_multilingual_vulnerability_dataset
    fetch_pku_safe_rlhf_dataset
    fetch_seclists_bias_testing_dataset
    fetch_sosbench_dataset
    fetch_tdc23_redteaming_dataset
    fetch_wmdp_dataset
    fetch_xstest_dataset



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
    pyrit_custom_result_retry
    pyrit_json_retry
    pyrit_target_retry
    pyrit_placeholder_retry
    RateLimitException
    remove_markdown_json

:py:mod:`pyrit.executor.attack`
===============================

.. automodule:: pyrit.executor.attack
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AttackAdversarialConfig
    AttackContext
    AttackConverterConfig
    AttackExecutor
    AttackScoringConfig
    AttackStrategy
    ContextComplianceAttack
    ConversationSession
    CrescendoAttack
    FlipAttack
    ManyShotJailbreakAttack
    MultiPromptSendingAttack
    MultiPromptSendingAttackContext
    MultiTurnAttackContext
    PromptSendingAttack
    RTASystemPromptPaths
    RedTeamingAttack
    RolePlayAttack
    SingleTurnAttackContext
    TAPAttack
    TAPAttackContext
    TAPAttackResult
    TreeOfAttacksWithPruningAttack
    SkeletonKeyAttack
    ConsoleAttackResultPrinter

:py:mod:`pyrit.executor.promptgen`
==================================

.. automodule:: pyrit.executor.promptgen
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AnecdoctorContext
    AnecdoctorGenerator
    AnecdoctorResult
    FuzzerContext
    FuzzerResult
    FuzzerGenerator
    FuzzerResultPrinter
    PromptGeneratorStrategy
    PromptGeneratorStrategyContext
    PromptGeneratorStrategyResult

:py:mod:`pyrit.executor.workflow`
=================================

.. automodule:: pyrit.executor.workflow
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    XPIAContext
    XPIAResult
    XPIAWorkflow
    XPIATestWorkflow
    XPIAManualProcessingWorkflow
    XPIAProcessingCallback
    XPIAStatus

:py:mod:`pyrit.memory`
======================

.. automodule:: pyrit.memory
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AttackResultEntry
    AzureSQLMemory
    CentralMemory
    EmbeddingDataEntry
    MemoryInterface
    MemoryEmbedding
    MemoryExporter
    PromptMemoryEntry
    SeedPromptEntry
    SQLiteMemory

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
    ConversationReference
    ConversationType
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
    StrategyResult
    TextDataTypeSerializer
    UnvalidatedScore


:py:mod:`pyrit.prompt_converter`
================================

.. automodule:: pyrit.prompt_converter
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AddImageTextConverter
    AddImageVideoConverter
    AddTextImageConverter
    AnsiAttackConverter
    AsciiArtConverter
    AsciiSmugglerConverter
    AtbashConverter
    AudioFrequencyConverter
    AzureSpeechAudioToTextConverter
    AzureSpeechTextToAudioConverter
    Base64Converter
    BinaryConverter
    CaesarConverter
    CharacterSpaceConverter
    CharSwapConverter
    CodeChameleonConverter
    ColloquialWordswapConverter
    ConverterResult
    DenylistConverter
    DiacriticConverter
    EmojiConverter
    FirstLetterConverter
    FlipConverter
    FuzzerCrossOverConverter
    FuzzerExpandConverter
    FuzzerRephraseConverter
    FuzzerShortenConverter
    FuzzerSimilarConverter
    HumanInTheLoopConverter
    ImageCompressionConverter
    InsertPunctuationConverter
    LeetspeakConverter
    LLMGenericTextConverter
    MaliciousQuestionGeneratorConverter
    MathPromptConverter
    MorseConverter
    NoiseConverter
    PDFConverter
    PersuasionConverter
    PromptConverter
    QRCodeConverter
    RandomCapitalLettersConverter
    RepeatTokenConverter
    ROT13Converter
    SearchReplaceConverter
    SneakyBitsSmugglerConverter
    StringJoinConverter
    SuffixAppendConverter
    SuperscriptConverter
    TemplateSegmentConverter
    TenseConverter
    TextJailbreakConverter
    TextToHexConverter
    ToneConverter
    ToxicSentenceGeneratorConverter
    TranslationConverter
    UnicodeConfusableConverter
    UnicodeReplacementConverter
    UnicodeSubstitutionConverter
    UrlConverter
    VariationConverter
    VariationSelectorSmugglerConverter
    ZalgoConverter
    ZeroWidthConverter

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
    PromptConverterConfiguration
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
    get_http_target_json_response_callback_function
    get_http_target_regex_matching_callback_function
    HTTPTarget
    HTTPXAPITarget
    HuggingFaceChatTarget
    HuggingFaceEndpointTarget
    limit_requests_per_minute
    OpenAICompletionTarget
    OpenAIDALLETarget
    OpenAIChatTarget
    OpenAIResponseTarget
    OpenAISoraTarget
    OpenAITTSTarget
    OpenAITarget
    PlaywrightTarget
    PromptChatTarget
    PromptShieldTarget
    PromptTarget
    RealtimeTarget
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
    BatchScorer
    ContentClassifierPaths
    CompositeScorer
    FloatScaleThresholdScorer
    GandalfScorer
    HarmHumanLabeledEntry
    HarmScorerEvaluator
    HarmScorerMetrics
    HumanInTheLoopScorer
    HumanInTheLoopScorerGradio
    HumanLabeledDataset
    HumanLabeledEntry
    InsecureCodeScorer
    LikertScalePaths
    LookBackScorer
    MarkdownInjectionScorer
    MetricsType
    ObjectiveHumanLabeledEntry
    ObjectiveScorerEvaluator
    ObjectiveScorerMetrics
    PlagiarismScorer
    PromptShieldScorer
    QuestionAnswerScorer
    Scorer
    ScorerEvaluator
    ScorerMetrics
    ScoreAggregator
    SelfAskCategoryScorer
    SelfAskGeneralScorer
    SelfAskLikertScorer
    SelfAskRefusalScorer
    SelfAskScaleScorer
    SelfAskTrueFalseScorer
    SelfAskQuestionAnswerScorer
    SubStringScorer
    TrueFalseInverterScorer
    TrueFalseQuestion
    TrueFalseQuestionPaths
