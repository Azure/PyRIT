import { useEffect, useMemo, useState } from 'react'
import {
  Spinner,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import AttackOptionList from './AttackOptionList'
import PromptOutputPanel from './PromptOutputPanel'
import TestDetailsPanel from './TestDetailsPanel'
import {
  getPrimaryInputType,
  buildPreviewParams,
  buildPromptPreview,
  getMissingRequiredParams,
  humanizeOptionName,
  humanizeParameterLabel,
  getSourceFieldLabel,
  isGoodForVideoPromptTesting,
  isVideoSpecificOption,
  worksWithImageUploads,
} from './builderUtils'
import {
  converterPreviewApi,
  converterTypesApi,
  targetsApi,
} from '../../services/api'
import type {
  ConverterPreviewResponse,
  ConverterTypeMetadata,
  PromptBuilderFormState,
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
})

const emptyFormState: PromptBuilderFormState = {
  selectedTargetId: '',
  sourceContent: '',
  parameterValues: {},
}

export default function PromptBuilderPage() {
  const styles = useStyles()
  const [searchValue, setSearchValue] = useState('')
  const [formState, setFormState] = useState<PromptBuilderFormState>(emptyFormState)
  const [options, setOptions] = useState<ConverterTypeMetadata[]>([])
  const [targets, setTargets] = useState<TargetInstance[]>([])
  const [selectedOptionType, setSelectedOptionType] = useState<string | null>(null)
  const [preview, setPreview] = useState<ConverterPreviewResponse | null>(null)
  const [previewError, setPreviewError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isPreviewLoading, setIsPreviewLoading] = useState(false)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const [typeResponse, targetResponse] = await Promise.all([
          converterTypesApi.listTypes(),
          targetsApi.listTargets(),
        ])

        if (cancelled) {
          return
        }

        setOptions(typeResponse.items)
        setTargets(targetResponse.items)

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
    setPreview(null)
    setPreviewError(null)
  }

  const handleParameterChange = (name: string, value: string | number | boolean) => {
    setFormState(current => ({
      ...current,
      parameterValues: {
        ...current.parameterValues,
        [name]: value,
      },
    }))
    setPreview(null)
    setPreviewError(null)
  }

  const handleSelectOption = (option: ConverterTypeMetadata) => {
    setSelectedOptionType(option.converter_type)
    setFormState(current => ({
      ...current,
      parameterValues: {},
    }))
    setPreview(null)
    setPreviewError(null)
  }

  const handleBuildPreview = async () => {
    if (!selectedOption) {
      return
    }

    setIsPreviewLoading(true)
    setPreviewError(null)

    try {
      const response = await converterPreviewApi.previewType(
        selectedOption.converter_type,
        buildPreviewParams(selectedOption, formState.parameterValues),
        basePrompt,
        getPrimaryInputType(selectedOption),
      )
      setPreview(response)
    } catch (error) {
      setPreview(null)
      setPreviewError(
        error instanceof Error
          ? error.message
          : 'The transformed prompt could not be built with the current settings.',
      )
    } finally {
      setIsPreviewLoading(false)
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
            setPreview(null)
            setPreviewError(null)
          }}
        />

        <PromptOutputPanel
          option={selectedOption}
          basePrompt={basePrompt}
          isPromptReady={Boolean(formState.sourceContent.trim())}
          preview={preview}
          isPreviewLoading={isPreviewLoading}
          previewError={previewError}
          canPreview={canPreview}
          previewHint={previewHint}
          onBuildTransformedPrompt={handleBuildPreview}
          onCopy={handleCopy}
        />
      </div>
    </div>
  )
}
