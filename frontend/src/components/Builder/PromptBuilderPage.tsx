import { useEffect, useMemo, useState } from 'react'
import {
  Spinner,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import AttackOptionList from './AttackOptionList'
import AttackStarterPanel from './AttackStarterPanel'
import PromptOutputPanel from './PromptOutputPanel'
import TestDetailsPanel from './TestDetailsPanel'
import {
  canRequestVariants,
  canUseAttackStarter,
  formatBlockedWords,
  getInitialPresetValues,
  getPrimaryInputType,
  buildPromptPreview,
  buildPreviewParams,
  expandPresetTemplate,
  getMissingRequiredParams,
  humanizeOptionName,
  humanizeParameterLabel,
  parseBlockedWords,
  getSourceFieldLabel,
  isGoodForVideoPromptTesting,
  isVideoSpecificOption,
  worksWithImageUploads,
} from './builderUtils'
import {
  builderApi,
  converterTypesApi,
  targetsApi,
} from '../../services/api'
import type {
  BuilderBuildResponse,
  BuilderConfigResponse,
  PromptBankPreset,
  ConverterTypeMetadata,
  PromptBuilderFormState,
  ReferenceImageResponse,
  TargetInstance,
} from '../../types'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    width: '100%',
    overflow: 'hidden',
    backgroundColor: tokens.colorNeutralBackground2,
  },
  ribbon: {
    borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
    backgroundColor: tokens.colorNeutralBackground1,
    padding: `${tokens.spacingVerticalM} ${tokens.spacingHorizontalXL}`,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  helper: {
    color: tokens.colorNeutralForeground3,
  },
  status: {
    color: tokens.colorBrandForeground1,
  },
  content: {
    flex: 1,
    minHeight: 0,
    display: 'grid',
    gridTemplateColumns: 'minmax(280px, 340px) minmax(340px, 1fr) minmax(340px, 1fr)',
    gridTemplateRows: 'minmax(0, 1fr)',
    gap: tokens.spacingHorizontalL,
    padding: tokens.spacingHorizontalXL,
    overflow: 'hidden',
    '@media (max-width: 1280px)': {
      gridTemplateColumns: '1fr',
      gridTemplateRows: 'none',
      overflowY: 'auto',
    },
  },
  loading: {
    flex: 1,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  middleColumn: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
    minHeight: 0,
    overflowY: 'auto',
    overflowX: 'hidden',
  },
})

const emptyFormState: PromptBuilderFormState = {
  selectedTargetId: '',
  sourceContent: '',
  selectedPresetId: '',
  presetValues: {},
  avoidBlockedWords: false,
  blockedWordsText: '',
  variantCount: 1,
  parameterValues: {},
}

export default function PromptBuilderPage() {
  const styles = useStyles()
  const [searchValue, setSearchValue] = useState('')
  const [formState, setFormState] = useState<PromptBuilderFormState>(emptyFormState)
  const [options, setOptions] = useState<ConverterTypeMetadata[]>([])
  const [targets, setTargets] = useState<TargetInstance[]>([])
  const [builderConfig, setBuilderConfig] = useState<BuilderConfigResponse | null>(null)
  const [selectedFamilyId, setSelectedFamilyId] = useState('')
  const [selectedOptionType, setSelectedOptionType] = useState<string | null>(null)
  const [buildResponse, setBuildResponse] = useState<BuilderBuildResponse | null>(null)
  const [selectedVariantId, setSelectedVariantId] = useState<string | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)
  const [referenceImage, setReferenceImage] = useState<ReferenceImageResponse | null>(null)
  const [latestGeneratedImage, setLatestGeneratedImage] = useState<ReferenceImageResponse | null>(null)
  const [isReferenceImageLoading, setIsReferenceImageLoading] = useState(false)
  const [referenceImageError, setReferenceImageError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const [typeResponse, targetResponse, builderConfigResponse] = await Promise.all([
          converterTypesApi.listTypes(),
          targetsApi.listTargets(),
          builderApi.getConfig(),
        ])

        if (cancelled) {
          return
        }

        setOptions(typeResponse.items)
        setTargets(targetResponse.items)
        setBuilderConfig(builderConfigResponse)
        const defaultFamilyId = builderConfigResponse.families[0]?.family_id || ''
        const defaultPreset = builderConfigResponse.presets.find(preset => preset.family_id === defaultFamilyId) || null
        setSelectedFamilyId(defaultFamilyId)
        setFormState(current => ({
          ...current,
          selectedPresetId: defaultPreset?.preset_id || '',
          presetValues: getInitialPresetValues(defaultPreset),
          blockedWordsText: formatBlockedWords(builderConfigResponse.defaults.default_blocked_words),
        }))

        if (typeResponse.items.length > 0) {
          const preferredDefault =
            typeResponse.items.find(isVideoSpecificOption) ||
            typeResponse.items.find(worksWithImageUploads) ||
            typeResponse.items.find(isGoodForVideoPromptTesting) ||
            typeResponse.items[0]

          setSelectedOptionType(preferredDefault.converter_type)
        }
      } catch (_error) {
        if (!cancelled) {
          setPreviewError('The builder could not load the current options. Check that the backend is running.')
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false)
        }
      }
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [])

  const selectedOption = useMemo(
    () => options.find(option => option.converter_type === selectedOptionType) || null,
    [options, selectedOptionType],
  )

  const selectedPreset = useMemo<PromptBankPreset | null>(() => {
    if (!builderConfig || !formState.selectedPresetId) {
      return null
    }

    return builderConfig.presets.find(preset => preset.preset_id === formState.selectedPresetId) || null
  }, [builderConfig, formState.selectedPresetId])

  const selectedVariant = useMemo(() => {
    if (!buildResponse) {
      return null
    }

    return buildResponse.variants.find(variant => variant.variant_id === selectedVariantId) || buildResponse.variants[0] || null
  }, [buildResponse, selectedVariantId])

  const basePrompt = useMemo(
    () => buildPromptPreview(formState),
    [formState],
  )

  const missingRequiredParams = useMemo(
    () => getMissingRequiredParams(selectedOption, formState.parameterValues),
    [formState.parameterValues, selectedOption],
  )

  const previewHint = useMemo(() => {
    if (!selectedOption) {
      return 'Choose an option to build its transformed prompt.'
    }

    if (!selectedOption.preview_supported) {
      return selectedOption.preview_unavailable_reason || 'This option needs extra setup before it can be previewed.'
    }

    if (!formState.sourceContent.trim()) {
      return `Add ${getSourceFieldLabel(selectedOption).toLowerCase()} so PyRIT has starting content to transform.`
    }

    if (missingRequiredParams.length > 0) {
      return `Add the required option settings: ${missingRequiredParams.map(param => humanizeParameterLabel(param)).join(', ')}.`
    }

    return 'Build transformed output to see the final result for this option.'
  }, [formState.sourceContent, missingRequiredParams, selectedOption])

  const canPreview = Boolean(
    selectedOption &&
      selectedOption.preview_supported &&
      formState.sourceContent.trim() &&
      missingRequiredParams.length === 0,
  )

  const handleFieldChange = (
    field: keyof Omit<PromptBuilderFormState, 'parameterValues'>,
    value: string,
  ) => {
    setFormState(current => ({
      ...current,
      [field]: value,
    }))
    setBuildResponse(null)
    setSelectedVariantId(null)
    setPreviewError(null)
    setReferenceImage(null)
    setReferenceImageError(null)
  }

  const handleParameterChange = (name: string, value: string | number | boolean) => {
    setFormState(current => ({
      ...current,
      parameterValues: {
        ...current.parameterValues,
        [name]: value,
      },
    }))
    setBuildResponse(null)
    setSelectedVariantId(null)
    setPreviewError(null)
    setReferenceImage(null)
    setReferenceImageError(null)
  }

  const handleSelectOption = (option: ConverterTypeMetadata) => {
    setSelectedOptionType(option.converter_type)
    setFormState(current => ({
      ...current,
      parameterValues: {},
      variantCount: canRequestVariants(option, builderConfig) ? current.variantCount : 1,
    }))
    setBuildResponse(null)
    setSelectedVariantId(null)
    setPreviewError(null)
    setReferenceImage(null)
    setReferenceImageError(null)
  }

  const handleBuildPreview = async () => {
    if (!selectedOption) {
      return
    }

    setIsPreviewLoading(true)
    setPreviewError(null)

    try {
      const response = await builderApi.build({
        source_content: formState.sourceContent,
        source_content_data_type: getPrimaryInputType(selectedOption),
        converter_type: selectedOption.converter_type,
        converter_params: buildPreviewParams(selectedOption, formState.parameterValues),
        preset_id: formState.selectedPresetId || null,
        preset_values: formState.presetValues,
        avoid_blocked_words: canUseAttackStarter(selectedOption) ? formState.avoidBlockedWords : false,
        blocked_words: parseBlockedWords(formState.blockedWordsText),
        variant_count: canRequestVariants(selectedOption, builderConfig) ? formState.variantCount : 1,
      })
      setBuildResponse(response)
      setSelectedVariantId(response.variants[0]?.variant_id || null)
      setReferenceImage(null)
      setReferenceImageError(null)
    } catch (error) {
      setBuildResponse(null)
      setSelectedVariantId(null)
      setPreviewError(
        error instanceof Error
          ? error.message
          : 'The transformed prompt could not be built with the current settings.',
      )
    } finally {
      setIsPreviewLoading(false)
    }
  }

  const handleGenerateReferenceImage = async () => {
    if (!selectedVariant || selectedVariant.data_type !== 'text') {
      return
    }

    setIsReferenceImageLoading(true)
    setReferenceImageError(null)

    try {
      const response = await builderApi.generateReferenceImage(selectedVariant.value)
      setReferenceImage(response)
      setLatestGeneratedImage(response)
    } catch (error) {
      setReferenceImage(null)
      setReferenceImageError(
        error instanceof Error
          ? error.message
          : 'The reference image could not be generated with the current helper model.',
      )
    } finally {
      setIsReferenceImageLoading(false)
    }
  }

  const handleCopy = async (value: string) => {
    if (!navigator.clipboard) {
      setPreviewError('Clipboard copy is not available in this browser.')
      return
    }

    try {
      await navigator.clipboard.writeText(value)
      setPreviewError(null)
    } catch (_error) {
      setPreviewError('Copy failed. You can still select the text and copy it manually.')
    }
  }

  const handlePresetChange = (presetId: string) => {
    const preset = builderConfig?.presets.find(item => item.preset_id === presetId) || null
    setFormState(current => ({
      ...current,
      selectedPresetId: presetId,
      presetValues: getInitialPresetValues(preset),
    }))
    setBuildResponse(null)
    setSelectedVariantId(null)
    setPreviewError(null)
    setReferenceImage(null)
    setReferenceImageError(null)
  }

  const handleApplyPreset = () => {
    const expanded = expandPresetTemplate(selectedPreset, formState.presetValues)
    if (!expanded) {
      return
    }

    setFormState(current => ({
      ...current,
      sourceContent: expanded,
    }))
    setBuildResponse(null)
    setSelectedVariantId(null)
    setPreviewError(null)
    setReferenceImage(null)
    setReferenceImageError(null)
  }

  if (isLoading) {
    return (
      <div className={styles.loading}>
        <Spinner label="Loading prompt builder" />
      </div>
    )
  }

  return (
    <div className={styles.root}>
      <div className={styles.ribbon}>
        <Text as="h1" size={600} weight="semibold">Prompt builder</Text>
        <Text className={styles.helper} block>
          Pick any available option, start from source content, and build the exact transformed result.
        </Text>
        <Text className={styles.status} block>
          {selectedOption ? `Selected option: ${humanizeOptionName(selectedOption)}` : 'Pick an option to get started'}
        </Text>
      </div>

      <div className={styles.content}>
        <AttackOptionList
          options={options}
          searchValue={searchValue}
          selectedConverterType={selectedOptionType}
          onSearchChange={setSearchValue}
          onSelect={handleSelectOption}
        />

        <div className={styles.middleColumn}>
          <AttackStarterPanel
            option={selectedOption}
            config={builderConfig}
            formState={formState}
            selectedFamilyId={selectedFamilyId}
            selectedPreset={selectedPreset}
            latestGeneratedImage={latestGeneratedImage}
            onFamilyChange={value => {
              setSelectedFamilyId(value)
              const nextPreset = builderConfig?.presets.find(preset => preset.family_id === value) || null
              setFormState(current => ({
                ...current,
                selectedPresetId: nextPreset?.preset_id || '',
                presetValues: getInitialPresetValues(nextPreset),
              }))
              setBuildResponse(null)
              setSelectedVariantId(null)
              setPreviewError(null)
              setReferenceImage(null)
              setReferenceImageError(null)
            }}
            onPresetChange={handlePresetChange}
            onPresetFieldChange={(name, value) => {
              setFormState(current => ({
                ...current,
                presetValues: {
                  ...current.presetValues,
                  [name]: value,
                },
              }))
              setBuildResponse(null)
              setSelectedVariantId(null)
              setPreviewError(null)
              setReferenceImage(null)
              setReferenceImageError(null)
            }}
            onApplyPreset={handleApplyPreset}
            onAvoidBlockedWordsChange={value => {
              setFormState(current => ({
                ...current,
                avoidBlockedWords: value,
              }))
              setBuildResponse(null)
              setSelectedVariantId(null)
              setPreviewError(null)
              setReferenceImage(null)
              setReferenceImageError(null)
            }}
            onBlockedWordsTextChange={value => {
              setFormState(current => ({
                ...current,
                blockedWordsText: value,
              }))
              setBuildResponse(null)
              setSelectedVariantId(null)
              setPreviewError(null)
              setReferenceImage(null)
              setReferenceImageError(null)
            }}
            onVariantCountChange={value => {
              setFormState(current => ({
                ...current,
                variantCount: value,
              }))
              setBuildResponse(null)
              setSelectedVariantId(null)
              setPreviewError(null)
              setReferenceImage(null)
              setReferenceImageError(null)
            }}
            onUseLatestGeneratedImage={() => {
              if (!latestGeneratedImage) {
                return
              }
              handleFieldChange('sourceContent', latestGeneratedImage.image_path)
            }}
          />

          <TestDetailsPanel
            option={selectedOption}
            targets={targets}
            formState={formState}
            onFieldChange={handleFieldChange}
            onParameterChange={handleParameterChange}
            onClearOptionSettings={() => {
              setFormState(current => ({
                ...current,
                parameterValues: {},
              }))
              setBuildResponse(null)
              setSelectedVariantId(null)
              setPreviewError(null)
            }}
          />
        </div>

        <PromptOutputPanel
          option={selectedOption}
          basePrompt={basePrompt}
          isPromptReady={Boolean(formState.sourceContent.trim())}
          buildResponse={buildResponse}
          selectedVariantId={selectedVariantId}
          isPreviewLoading={isPreviewLoading}
          previewError={previewError}
          canPreview={canPreview}
          previewHint={previewHint}
          onBuildTransformedPrompt={handleBuildPreview}
          onSelectVariant={setSelectedVariantId}
          onCopy={handleCopy}
          referenceImage={referenceImage}
          referenceImageAvailable={Boolean(builderConfig?.capabilities.reference_image_available)}
          referenceImageTargetName={builderConfig?.capabilities.reference_image_target_name || null}
          isReferenceImageLoading={isReferenceImageLoading}
          referenceImageError={referenceImageError}
          onGenerateReferenceImage={handleGenerateReferenceImage}
        />
      </div>
    </div>
  )
}
