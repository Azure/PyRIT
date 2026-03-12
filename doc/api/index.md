# API Reference

## [pyrit.analytics](pyrit_analytics.md)

`ApproximateTextMatching`, `AttackStats`, `ConversationAnalytics`, `ExactTextMatching`, `TextMatching` ... (6 total)

## [pyrit.auth](pyrit_auth.md)

`Authenticator`, `AzureAuth`, `AzureStorageAuth`, `CopilotAuthenticator`, `ManualCopilotAuthenticator`, `TokenProviderCredential` ... (10 total)

## [pyrit.common](pyrit_common.md)

`DefaultValueScope`, `Singleton`, `YamlLoadable` ... (26 total)

## [pyrit.datasets](pyrit_datasets.md)

 ... (4 total)

## [pyrit.embedding](pyrit_embedding.md)

`OpenAITextEmbedding`

## [pyrit.exceptions](pyrit_exceptions.md)

`BadRequestException`, `ComponentRole`, `EmptyResponseException`, `ExecutionContext`, `ExecutionContextManager`, `InvalidJsonException`, `MissingPromptPlaceholderException`, `PyritException` ... (20 total)

## [pyrit.executor.attack.component](pyrit_executor_attack_component.md)

`ConversationManager`, `ConversationState`, `PrependedConversationConfig` ... (7 total)

## [pyrit.executor.attack.core](pyrit_executor_attack_core.md)

`AttackAdversarialConfig`, `AttackContext`, `AttackConverterConfig`, `AttackExecutor`, `AttackExecutorResult`, `AttackParameters`, `AttackScoringConfig`, `AttackStrategy`

## [pyrit.executor.attack](pyrit_executor_attack.md)

`AttackAdversarialConfig`, `AttackContext`, `AttackConverterConfig`, `AttackExecutor`, `AttackExecutorResult`, `AttackParameters`, `AttackResultPrinter`, `AttackScoringConfig` ... (39 total)

## [pyrit.executor.attack.multi_turn](pyrit_executor_attack_multi_turn.md)

`ChunkedRequestAttack`, `ChunkedRequestAttackContext`, `ConversationSession`, `CrescendoAttack`, `CrescendoAttackContext`, `CrescendoAttackResult`, `MultiPromptSendingAttack`, `MultiPromptSendingAttackParameters` ... (16 total)

## [pyrit.executor.attack.printer](pyrit_executor_attack_printer.md)

`AttackResultPrinter`, `ConsoleAttackResultPrinter`, `MarkdownAttackResultPrinter`

## [pyrit.executor.attack.single_turn](pyrit_executor_attack_single_turn.md)

`ContextComplianceAttack`, `FlipAttack`, `ManyShotJailbreakAttack`, `PromptSendingAttack`, `RolePlayAttack`, `RolePlayPaths`, `SingleTurnAttackContext`, `SingleTurnAttackStrategy` ... (9 total)

## [pyrit.executor.benchmark](pyrit_executor_benchmark.md)

`FairnessBiasBenchmark`, `FairnessBiasBenchmarkContext`, `QuestionAnsweringBenchmark`, `QuestionAnsweringBenchmarkContext`

## [pyrit.executor.core](pyrit_executor_core.md)

`Strategy`, `StrategyContext`, `StrategyConverterConfig`, `StrategyEvent`, `StrategyEventData`, `StrategyEventHandler`

## [pyrit.executor](pyrit_executor.md)

## [pyrit.executor.promptgen.core](pyrit_executor_promptgen_core.md)

`PromptGeneratorStrategy`, `PromptGeneratorStrategyContext`, `PromptGeneratorStrategyResult`

## [pyrit.executor.promptgen.fuzzer](pyrit_executor_promptgen_fuzzer.md)

`FuzzerContext`, `FuzzerConverter`, `FuzzerCrossOverConverter`, `FuzzerExpandConverter`, `FuzzerGenerator`, `FuzzerRephraseConverter`, `FuzzerResult`, `FuzzerResultPrinter` ... (10 total)

## [pyrit.executor.promptgen](pyrit_executor_promptgen.md)

`AnecdoctorContext`, `AnecdoctorGenerator`, `AnecdoctorResult`, `PromptGeneratorStrategy`, `PromptGeneratorStrategyContext`, `PromptGeneratorStrategyResult`

## [pyrit.executor.workflow](pyrit_executor_workflow.md)

`XPIAContext`, `XPIAManualProcessingWorkflow`, `XPIAProcessingCallback`, `XPIAResult`, `XPIAStatus`, `XPIATestWorkflow`, `XPIAWorkflow`

## [pyrit.identifiers](pyrit_identifiers.md)

`AtomicAttackEvaluationIdentifier`, `ChildEvalRule`, `ComponentIdentifier`, `EvaluationIdentifier`, `Identifiable`, `ScorerEvaluationIdentifier` ... (12 total)

## [pyrit.memory](pyrit_memory.md)

`AttackResultEntry`, `AzureSQLMemory`, `CentralMemory`, `EmbeddingDataEntry`, `MemoryEmbedding`, `MemoryExporter`, `MemoryInterface`, `PromptMemoryEntry` ... (10 total)

## [pyrit.message_normalizer](pyrit_message_normalizer.md)

`ChatMessageNormalizer`, `ConversationContextNormalizer`, `GenericSystemSquashNormalizer`, `MessageListNormalizer`, `MessageStringNormalizer`, `TokenizerTemplateNormalizer`

## [pyrit.models](pyrit_models.md)

`AttackOutcome`, `AttackResult`, `AudioPathDataTypeSerializer`, `AzureBlobStorageIO`, `BinaryPathDataTypeSerializer`, `ChatMessage`, `ChatMessageListDictContent`, `ChatMessagesDataset` ... (50 total)

## [pyrit.models.seeds](pyrit_models_seeds.md)

`NextMessageSystemPromptPaths`, `Seed`, `SeedAttackGroup`, `SeedAttackTechniqueGroup`, `SeedDataset`, `SeedGroup`, `SeedObjective`, `SeedPrompt` ... (10 total)

## [pyrit.prompt_converter](pyrit_prompt_converter.md)

`AddImageTextConverter`, `AddImageVideoConverter`, `AddTextImageConverter`, `AllWordsSelectionStrategy`, `AsciiArtConverter`, `AsciiSmugglerConverter`, `AskToDecodeConverter`, `AtbashConverter` ... (92 total)

## [pyrit.prompt_converter.token_smuggling](pyrit_prompt_converter_token_smuggling.md)

`AsciiSmugglerConverter`, `SneakyBitsSmugglerConverter`, `VariationSelectorSmugglerConverter`

## [pyrit.prompt_normalizer](pyrit_prompt_normalizer.md)

`NormalizerRequest`, `PromptConverterConfiguration`, `PromptNormalizer`

## [pyrit.prompt_target](pyrit_prompt_target.md)

`AzureBlobStorageTarget`, `AzureMLChatTarget`, `CopilotType`, `CrucibleTarget`, `GandalfLevel`, `GandalfTarget`, `PlaywrightCopilotTarget`, `PlaywrightTarget` ... (30 total)

## [pyrit.registry.class_registries](pyrit_registry_class_registries.md)

`BaseClassRegistry`, `ClassEntry`, `InitializerMetadata`, `InitializerRegistry`, `ScenarioMetadata`, `ScenarioRegistry`

## [pyrit.registry.instance_registries](pyrit_registry_instance_registries.md)

`BaseInstanceRegistry`, `ConverterRegistry`, `ScorerRegistry`, `TargetRegistry`

## [pyrit.registry](pyrit_registry.md)

`BaseClassRegistry`, `BaseInstanceRegistry`, `ClassEntry`, `InitializerMetadata`, `InitializerRegistry`, `RegistryProtocol`, `ScenarioMetadata`, `ScenarioRegistry` ... (13 total)

## [pyrit.scenario.core](pyrit_scenario_core.md)

`AtomicAttack`, `DatasetConfiguration`, `Scenario`, `ScenarioCompositeStrategy`, `ScenarioStrategy`

## [pyrit.scenario](pyrit_scenario.md)

`AtomicAttack`, `DatasetConfiguration`, `Scenario`, `ScenarioCompositeStrategy`, `ScenarioIdentifier`, `ScenarioResult`, `ScenarioStrategy` ... (8 total)

## [pyrit.scenario.printer](pyrit_scenario_printer.md)

`ConsoleScenarioResultPrinter`, `ScenarioResultPrinter`

## [pyrit.scenario.scenarios.airt](pyrit_scenario_scenarios_airt.md)

`ContentHarms`, `ContentHarmsStrategy`, `Cyber`, `CyberStrategy`, `Jailbreak`, `JailbreakStrategy`, `Leakage`, `LeakageScenario` ... (14 total)

## [pyrit.scenario.scenarios.foundry](pyrit_scenario_scenarios_foundry.md)

`FoundryScenario`, `FoundryStrategy`, `RedTeamAgent`

## [pyrit.scenario.scenarios.garak](pyrit_scenario_scenarios_garak.md)

`Encoding`, `EncodingStrategy`

## [pyrit.score](pyrit_score.md)

`BatchScorer`, `ConsoleScorerPrinter`, `ConversationScorer`, `Scorer`, `ScorerPrinter`, `ScorerPromptValidator` ... (62 total)

## [pyrit.score.printer](pyrit_score_printer.md)

`ConsoleScorerPrinter`, `ScorerPrinter`

## [pyrit.setup.initializers.components](pyrit_setup_initializers_components.md)

`ScorerInitializer`, `TargetConfig`, `TargetInitializer`

## [pyrit.setup.initializers](pyrit_setup_initializers.md)

`AIRTInitializer`, `LoadDefaultDatasets`, `PyRITInitializer`, `ScenarioObjectiveListInitializer`, `ScenarioObjectiveTargetInitializer`, `ScorerInitializer`, `SimpleInitializer`, `TargetInitializer`

## [pyrit.setup](pyrit_setup.md)

`ConfigurationLoader` ... (3 total)
