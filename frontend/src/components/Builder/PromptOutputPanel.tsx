import {
  Badge,
  Button,
  Spinner,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import type { ConverterPreviewResponse, ConverterTypeMetadata } from '../../types'
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
  buttonRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalS,
  },
})

interface PromptOutputPanelProps {
  option: ConverterTypeMetadata | null
  basePrompt: string
  isPromptReady: boolean
  preview: ConverterPreviewResponse | null
  isPreviewLoading: boolean
  previewError: string | null
  canPreview: boolean
  previewHint: string
  onBuildTransformedPrompt: () => void
  onCopy: (value: string) => void
}

export default function PromptOutputPanel({
  option,
  basePrompt,
  isPromptReady,
  preview,
  isPreviewLoading,
  previewError,
  canPreview,
  previewHint,
  onBuildTransformedPrompt,
  onCopy,
}: PromptOutputPanelProps) {
  const styles = useStyles()
  const sourceLabel = option ? `${humanizeDataType(getPrimaryInputType(option))} input` : 'Source input'
  const transformedLabel = preview
    ? `${humanizeDataType(preview.converted_value_data_type)} output`
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
          <pre className={styles.codeBlock}>{basePrompt}</pre>
        </div>

        <div className={styles.buttonRow}>
          <Button appearance="secondary" onClick={() => onCopy(basePrompt)}>
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

        {preview ? (
          <>
            <pre className={styles.codeBlock}>{preview.converted_value}</pre>
            <div className={styles.buttonRow}>
              <Button appearance="secondary" onClick={() => onCopy(preview.converted_value)}>
                Copy transformed output
              </Button>
            </div>
            <Text className={styles.helper} block>
              The selected option changed the input from {preview.original_value_data_type} to {preview.converted_value_data_type}.
            </Text>
          </>
        ) : (
          <Text className={styles.helper} block>
            Build the transformed output to see the exact result for this option.
          </Text>
        )}
      </div>
    </section>
  )
}
