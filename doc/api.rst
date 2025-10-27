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

    analyze_results
    AttackStats
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

    apply_defaults
    apply_defaults_to_method
    combine_dict
    combine_list
    convert_local_image_to_data_url
    DefaultValueScope
    deprecation_message
    display_image_response
    download_chunk
    download_file
    download_files
    download_specific_files
    get_available_files
    get_global_default_values
    get_httpx_client
    get_kwarg_param
    get_non_required_value
    get_random_indices
    get_required_value
    is_in_ipython_session
    make_request_and_raise_if_error_async
    print_chat_messages_with_color
    reset_default_values
    set_default_value
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
    fetch_harmbench_multimodal_dataset_async
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
    ImplicareAttack
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
    SeedEntry
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
    group_conversation_message_pieces_by_sequence
    Identifier
    ImagePathDataTypeSerializer
    Message
    MessagePiece
    PromptDataType
    PromptResponseError
    QuestionAnsweringDataset
    QuestionAnsweringEntry
    QuestionChoice
    Score
    ScoreType
    SeedPrompt
    SeedDataset
    SeedGroup
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
    BinAsciiConverter
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
    ImplicareConverter
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
    ToneConverter
    ToxicSentenceGeneratorConverter
    TranslationConverter
    TransparencyAttackConverter
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
    FloatScaleScorer
    FloatScaleThresholdScorer
    GandalfScorer
    HarmHumanLabeledEntry
    HarmScorerEvaluator
    HarmScorerMetrics
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
    PlagiarismMetric
    PlagiarismScorer
    PromptShieldScorer
    QuestionAnswerScorer
    Scorer
    ScorerEvaluator
    ScorerMetrics
    ScorerPromptValidator
    SelfAskCategoryScorer
    SelfAskGeneralFloatScaleScorer
    SelfAskGeneralTrueFalseScorer
    SelfAskLikertScorer
    SelfAskQuestionAnswerScorer
    SelfAskRefusalScorer
    SelfAskScaleScorer
    SelfAskTrueFalseScorer
    SubStringScorer
    TrueFalseCompositeScorer
    TrueFalseInverterScorer
    TrueFalseQuestion
    TrueFalseQuestionPaths
    TrueFalseScoreAggregator
    TrueFalseScorer

:py:mod:`pyrit.scenarios`
=========================

.. automodule:: pyrit.scenarios
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    AtomicAttack
    AtomicAttackResult
    EncodingScenario
    FoundryAttackStrategy
    FoundryScenario
    Scenario
    ScenarioAttackStrategy
    ScenarioIdentifier
    ScenarioResult

:py:mod:`pyrit.setup`
=====================

.. automodule:: pyrit.setup
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    initialize_pyrit
    AZURE_SQL
    SQLITE
    IN_MEMORY

:py:mod:`pyrit.setup.initializers`
==================================

.. automodule:: pyrit.setup.initializers
    :no-members:
    :no-inherited-members:

.. autosummary::
    :nosignatures:
    :toctree: _autosummary/

    PyRITInitializer
    AIRTInitializer
    SimpleInitializer
