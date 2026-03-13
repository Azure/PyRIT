import {
  Badge,
  Button,
  Spinner,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import type {
  BuilderBuildResponse,
  ConverterTypeMetadata,
  ReferenceImageResponse,
} from '../../types'
import {
  getVideoWorkflowGuidance,
  getPrimaryInputType,
  getPrimaryOutputType,
  humanizeDataType,
  humanizeOptionName,
} from './builderUtils'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
    minHeight: 0,
    overflowY: 'auto',
  },
  panel: {
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    borderRadius: tokens.borderRadiusLarge,
    backgroundColor: tokens.colorNeutralBackground1,
    padding: tokens.spacingHorizontalL,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
  },
  codeBlock: {
    margin: 0,
    padding: tokens.spacingHorizontalM,
    borderRadius: tokens.borderRadiusMedium,
    backgroundColor: tokens.colorNeutralBackground3,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    fontFamily: 'ui-monospace, SFMono-Regular, SFMono-Regular, Menlo, Consolas, monospace',
    fontSize: tokens.fontSizeBase200,
    lineHeight: tokens.lineHeightBase300,
  },
  helper: {
    color: tokens.colorNeutralForeground3,
  },
  stepList: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
  },
  stepCard: {
    borderRadius: tokens.borderRadiusMedium,
    backgroundColor: tokens.colorNeutralBackground2,
    padding: tokens.spacingHorizontalM,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  buttonRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalS,
  },
  variantRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalS,
  },
  selectedVariant: {
    boxShadow: `inset 0 0 0 1px ${tokens.colorBrandStroke1}`,
  },
  imagePreview: {
    width: '100%',
    maxHeight: '320px',
    objectFit: 'cover',
    borderRadius: tokens.borderRadiusMedium,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
  },
})

interface PromptOutputPanelProps {
  option: ConverterTypeMetadata | null
  basePrompt: string
  isPromptReady: boolean
  buildResponse: BuilderBuildResponse | null
  selectedVariantId: string | null
  isPreviewLoading: boolean
  previewError: string | null
  canPreview: boolean
  previewHint: string
  onBuildTransformedPrompt: () => void
  onSelectVariant: (variantId: string) => void
  onCopy: (value: string) => void
  referenceImage: ReferenceImageResponse | null
  referenceImageAvailable: boolean
  referenceImageTargetName: string | null
  isReferenceImageLoading: boolean
  referenceImageError: string | null
  onGenerateReferenceImage: () => void
}

export default function PromptOutputPanel({
  option,
  basePrompt,
  isPromptReady,
  buildResponse,
  selectedVariantId,
  isPreviewLoading,
  previewError,
  canPreview,
  previewHint,
  onBuildTransformedPrompt,
  onSelectVariant,
  onCopy,
  referenceImage,
  referenceImageAvailable,
  referenceImageTargetName,
  isReferenceImageLoading,
  referenceImageError,
  onGenerateReferenceImage,
}: PromptOutputPanelProps) {
  const styles = useStyles()
  const sourceLabel = option ? `${humanizeDataType(getPrimaryInputType(option))} input` : 'Source input'
  const displayedSourceValue = buildResponse?.resolved_source_value || basePrompt
  const selectedVariant = buildResponse?.variants.find(variant => variant.variant_id === selectedVariantId) || buildResponse?.variants[0] || null
  const transformedLabel = selectedVariant
    ? `${humanizeDataType(selectedVariant.data_type)} output`
    : option
      ? `${humanizeDataType(getPrimaryOutputType(option))} output`
      : 'Transformed output'

  return (
    <section className={styles.root}>
      <div className={styles.panel}>
        <div>
          <Text as="h2" size={500} weight="semibold">Built output</Text>
          <Text className={styles.helper} block>
            PyRIT starts from the source input below, then applies the selected option.
          </Text>
        </div>

        <div className={styles.buttonRow}>
          <Badge appearance="outline">{option ? humanizeOptionName(option) : 'No option selected'}</Badge>
          <Badge appearance="outline">{isPromptReady ? 'Input ready' : 'Needs more input'}</Badge>
          {option && <Badge appearance="outline">{getVideoWorkflowGuidance(option).title}</Badge>}
        </div>

        <div>
          <Text weight="semibold">{sourceLabel}</Text>
          <pre className={styles.codeBlock}>{displayedSourceValue}</pre>
        </div>

        <div className={styles.buttonRow}>
          <Button appearance="secondary" onClick={() => onCopy(displayedSourceValue)}>
            Copy source input
          </Button>
          <Button appearance="primary" onClick={onBuildTransformedPrompt} disabled={!canPreview || isPreviewLoading}>
            Build transformed output
          </Button>
        </div>

        <Text className={styles.helper} block>{previewHint}</Text>

        {isPreviewLoading && <Spinner label="Building transformed output" />}

        {previewError && (
          <Text block>
            {previewError}
          </Text>
        )}
      </div>

      <div className={styles.panel}>
        <Text as="h3" weight="semibold">{transformedLabel}</Text>

        {buildResponse ? (
          <>
            {buildResponse.variants.length > 1 && (
              <div className={styles.variantRow}>
                {buildResponse.variants.map(variant => (
                  <Button
                    key={variant.variant_id}
                    appearance={variant.variant_id === selectedVariant?.variant_id ? 'primary' : 'secondary'}
                    className={variant.variant_id === selectedVariant?.variant_id ? styles.selectedVariant : undefined}
                    onClick={() => onSelectVariant(variant.variant_id)}
                  >
                    {variant.label}
                  </Button>
                ))}
              </div>
            )}

            <pre className={styles.codeBlock}>{selectedVariant?.value || buildResponse.converted_value}</pre>
            <div className={styles.buttonRow}>
              <Button appearance="secondary" onClick={() => onCopy(selectedVariant?.value || buildResponse.converted_value)}>
                Copy transformed output
              </Button>
              {selectedVariant?.data_type === 'text' && (
                <Button
                  appearance="secondary"
                  onClick={onGenerateReferenceImage}
                  disabled={!referenceImageAvailable || isReferenceImageLoading}
                >
                  {referenceImage ? 'Regenerate reference image' : 'Generate reference image'}
                </Button>
              )}
            </div>
            <Text className={styles.helper} block>
              The selected option changed the input from {buildResponse.resolved_source_data_type} to {buildResponse.converted_value_data_type}.
            </Text>
            {buildResponse.warnings.length > 0 && buildResponse.warnings.map(warning => (
              <Text key={warning} className={styles.helper} block>{warning}</Text>
            ))}
            {buildResponse.steps.length > 0 && (
              <div className={styles.stepList}>
                <Text weight="semibold">Builder steps</Text>
                {buildResponse.steps.map(step => (
                  <div key={`${step.stage}-${step.title}`} className={styles.stepCard}>
                    <Text weight="semibold">{step.title}</Text>
                    {step.detail && <Text className={styles.helper} block>{step.detail}</Text>}
                    <Text className={styles.helper} block>
                      {humanizeDataType(step.input_data_type)} in {"->"} {humanizeDataType(step.output_data_type)} out
                    </Text>
                    <pre className={styles.codeBlock}>{step.output_value}</pre>
                  </div>
                ))}
              </div>
            )}
          </>
        ) : (
          <Text className={styles.helper} block>
            Build the transformed output to see the exact result for this option.
          </Text>
        )}
      </div>

      {(selectedVariant?.data_type === 'text' || referenceImage || referenceImageError) && (
        <div className={styles.panel}>
          <Text as="h3" weight="semibold">Reference image</Text>
          {referenceImageAvailable ? (
            <>
              <Text className={styles.helper} block>
                Generate an optional companion image from the selected text version.
                {referenceImageTargetName ? ` Helper target: ${referenceImageTargetName}.` : ''}
              </Text>
              {isReferenceImageLoading && <Spinner label="Generating reference image" />}
              {referenceImageError && <Text block>{referenceImageError}</Text>}
              {referenceImage && (
                <>
                  <img src={referenceImage.image_url} alt="Generated reference image" className={styles.imagePreview} />
                  <div className={styles.buttonRow}>
                    <Button appearance="secondary" onClick={() => onCopy(referenceImage.image_path)}>
                      Copy image path
                    </Button>
                  </div>
                </>
              )}
            </>
          ) : (
            <Text className={styles.helper} block>
              Reference image generation is unavailable until an image-capable helper target is configured.
            </Text>
          )}
        </div>
      )}
    </section>
  )
}
