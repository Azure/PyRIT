import {
  Badge,
  Button,
  Dropdown,
  Field,
  Input,
  Option,
  Text,
  Textarea,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import type {
  ConverterParameterMetadata,
  ConverterTypeMetadata,
  PromptBuilderFormState,
  TargetInstance,
} from '../../types'
import {
  formatOptionFlow,
  getVideoWorkflowGuidance,
  getPrimaryInputType,
  getSourceFieldHint,
  getSourceFieldLabel,
  getSourceFieldPlaceholder,
  humanizeOptionDescription,
  humanizeOptionName,
  humanizeParameterHint,
  humanizeParameterLabel,
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
  helper: {
    color: tokens.colorNeutralForeground3,
  },
  metaRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalXS,
  },
  guidanceBox: {
    borderRadius: tokens.borderRadiusMedium,
    backgroundColor: tokens.colorBrandBackground2,
    padding: tokens.spacingHorizontalM,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXXS,
  },
  select: {
    width: '100%',
    minHeight: '36px',
    borderRadius: tokens.borderRadiusMedium,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    backgroundColor: tokens.colorNeutralBackground1,
    color: tokens.colorNeutralForeground1,
    padding: `0 ${tokens.spacingHorizontalM}`,
    fontSize: tokens.fontSizeBase300,
  },
  details: {
    borderTop: `1px solid ${tokens.colorNeutralStroke2}`,
    paddingTop: tokens.spacingVerticalM,
  },
  detailsSummary: {
    cursor: 'pointer',
    fontWeight: tokens.fontWeightSemibold,
  },
  fieldGrid: {
    display: 'grid',
    gap: tokens.spacingVerticalM,
  },
  unsupportedList: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  noteBox: {
    borderRadius: tokens.borderRadiusMedium,
    backgroundColor: tokens.colorNeutralBackground2,
    padding: tokens.spacingHorizontalM,
  },
  actionRow: {
    display: 'flex',
    justifyContent: 'flex-start',
  },
})

const internalDefaultSettingNames = new Set([
  'converter_target',
  'prompt_template',
  'system_prompt_template',
  'user_prompt_template_with_objective',
])

const featuredOptionalSettingNames = new Set([
  'length_mode',
])

interface TestDetailsPanelProps {
  option: ConverterTypeMetadata | null
  targets: TargetInstance[]
  formState: PromptBuilderFormState
  onFieldChange: (
    field: keyof Omit<PromptBuilderFormState, 'parameterValues'>,
    value: string,
  ) => void
  onParameterChange: (name: string, value: string | number | boolean) => void
  onClearOptionSettings: () => void
}

function renderParameterControl(
  parameter: ConverterParameterMetadata,
  value: string | number | boolean | undefined,
  selectClassName: string,
  onChange: (value: string | number | boolean) => void,
) {
  const label = humanizeParameterLabel(parameter)
  const commonProps = {
    value: typeof value === 'boolean' ? String(value) : value === undefined ? '' : String(value),
  }
  const formatOptionLabel = (option: string) =>
    option
      .split('_')
      .filter(Boolean)
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ')

  if (parameter.input_kind === 'boolean') {
    return (
      <Dropdown
        className={selectClassName}
        aria-label={label}
        placeholder="Use default"
        selectedOptions={typeof value === 'boolean' ? [String(value)] : []}
        value={typeof value === 'boolean' ? (value ? 'True' : 'False') : ''}
        onOptionSelect={(_, data) => onChange(data.optionValue === 'true')}
      >
        <Option value="true" text="True">True</Option>
        <Option value="false" text="False">False</Option>
      </Dropdown>
    )
  }

  if (parameter.input_kind === 'select') {
    return (
      <Dropdown
        className={selectClassName}
        aria-label={label}
        placeholder="Select an option"
        selectedOptions={commonProps.value ? [commonProps.value] : []}
        value={commonProps.value ? formatOptionLabel(commonProps.value) : ''}
        onOptionSelect={(_, data) => onChange(data.optionValue || '')}
      >
        {(parameter.options || []).map(option => (
          <Option key={option} value={option} text={formatOptionLabel(option)}>
            {formatOptionLabel(option)}
          </Option>
        ))}
      </Dropdown>
    )
  }

  if (parameter.input_kind === 'list') {
    return (
      <Textarea
        aria-label={label}
        value={commonProps.value}
        onChange={(_, data) => onChange(data.value)}
        placeholder="Enter one item per line or separate with commas"
        rows={3}
      />
    )
  }

  const inputType = parameter.input_kind === 'number' ? 'number' : 'text'
  return (
    <Input
      aria-label={label}
      type={inputType}
      value={commonProps.value}
      onChange={(_, data) => onChange(data.value)}
      placeholder={parameter.default_value || humanizeParameterLabel(parameter)}
    />
  )
}

function groupParameters(
  parameters: ConverterParameterMetadata[],
): Record<'required' | 'optional' | 'unsupported', ConverterParameterMetadata[]> {
  return parameters.reduce(
    (groups, parameter) => {
      if (parameter.input_kind === 'unsupported') {
        groups.unsupported.push(parameter)
      } else if (parameter.required) {
        groups.required.push(parameter)
      } else {
        groups.optional.push(parameter)
      }
      return groups
    },
    {
      required: [] as ConverterParameterMetadata[],
      optional: [] as ConverterParameterMetadata[],
      unsupported: [] as ConverterParameterMetadata[],
    },
  )
}

export default function TestDetailsPanel({
  option,
  targets,
  formState,
  onFieldChange,
  onParameterChange,
  onClearOptionSettings,
}: TestDetailsPanelProps) {
  const styles = useStyles()

  if (!option) {
    return (
      <section className={styles.root}>
        <div className={styles.panel}>
          <Text as="h2" size={500} weight="semibold">Test details</Text>
          <Text className={styles.helper} block>
            Pick an option on the left to see what it means and what it needs.
          </Text>
        </div>
      </section>
    )
  }

  const groupedParameters = groupParameters(option.parameters)
  const workflowGuidance = getVideoWorkflowGuidance(option)
  const featuredOptionalSettings = groupedParameters.optional.filter(parameter =>
    featuredOptionalSettingNames.has(parameter.name),
  )
  const remainingOptionalSettings = groupedParameters.optional.filter(
    parameter => !featuredOptionalSettingNames.has(parameter.name),
  )
  const internalDefaultSettings = groupedParameters.unsupported.filter(
    parameter => !parameter.required && internalDefaultSettingNames.has(parameter.name),
  )
  const visibleUnsupportedSettings = groupedParameters.unsupported.filter(
    parameter => !(!parameter.required && internalDefaultSettingNames.has(parameter.name)),
  )

  return (
    <section className={styles.root}>
      <div className={styles.panel}>
        <div>
          <Text as="h2" size={500} weight="semibold">{humanizeOptionName(option)}</Text>
          <Text className={styles.helper} block>{humanizeOptionDescription(option)}</Text>
        </div>

        <div className={styles.metaRow}>
          <Badge appearance="outline">{formatOptionFlow(option)}</Badge>
          <Badge appearance="outline">{workflowGuidance.title}</Badge>
        </div>

        <div className={styles.guidanceBox}>
          <Text weight="semibold">{workflowGuidance.title}</Text>
          <Text className={styles.helper} block>
            {workflowGuidance.body}
          </Text>
        </div>

        <Field
          label="Specific system to test (optional)"
          hint={
            targets.length > 0
              ? 'Use this only when you want to tie the prompt to one concrete system.'
              : 'No concrete targets are loaded right now. You can still build prompts without picking one.'
          }
        >
          <Dropdown
            className={styles.select}
            aria-label="Specific system to test"
            placeholder="Choose a specific target"
            selectedOptions={formState.selectedTargetId ? [formState.selectedTargetId] : []}
            value={
              formState.selectedTargetId
                ? (() => {
                    const selectedTarget = targets.find(
                      target => target.target_registry_name === formState.selectedTargetId,
                    )
                    if (!selectedTarget) {
                      return formState.selectedTargetId
                    }
                    return `${selectedTarget.target_registry_name}${
                      selectedTarget.model_name ? ` (${selectedTarget.model_name})` : ''
                    }`
                  })()
                : ''
            }
            onOptionSelect={(_, data) => onFieldChange('selectedTargetId', data.optionValue || '')}
            disabled={targets.length === 0}
          >
            {targets.map(target => (
              <Option
                key={target.target_registry_name}
                value={target.target_registry_name}
                text={`${target.target_registry_name}${target.model_name ? ` (${target.model_name})` : ''}`}
              >
                {target.target_registry_name}
                {target.model_name ? ` (${target.model_name})` : ''}
              </Option>
            ))}
          </Dropdown>
        </Field>

        <Field
          label={getSourceFieldLabel(option)}
          hint={getSourceFieldHint(option)}
        >
          {getPrimaryInputType(option) === 'text' ? (
            <Textarea
              value={formState.sourceContent}
              onChange={(_, data) => onFieldChange('sourceContent', data.value)}
              placeholder={getSourceFieldPlaceholder(option)}
              rows={5}
            />
          ) : (
            <Input
              value={formState.sourceContent}
              onChange={(_, data) => onFieldChange('sourceContent', data.value)}
              placeholder={getSourceFieldPlaceholder(option)}
            />
          )}
        </Field>

        {['image_path', 'video_path', 'binary_path'].includes(getPrimaryInputType(option)) && (
          <div className={styles.noteBox}>
            <Text weight="semibold">Source file note</Text>
            <Text className={styles.helper} block>
              This builder currently expects a saved file path or stored URL here. It does not upload a new browser file from this panel yet.
            </Text>
          </div>
        )}
      </div>

      <div className={styles.panel}>
        <Text as="h3" weight="semibold">Option settings</Text>
        <Text className={styles.helper} block>
          All settings are listed here. Common ones stay open. Less common ones are collapsed, not hidden.
        </Text>

        <div className={styles.fieldGrid}>
          {groupedParameters.required.map(parameter => (
            <Field
              key={parameter.name}
              label={`${humanizeParameterLabel(parameter)}${parameter.required ? ' *' : ''}`}
              hint={humanizeParameterHint(parameter)}
            >
              {renderParameterControl(
                parameter,
                formState.parameterValues[parameter.name],
                styles.select,
                value => onParameterChange(parameter.name, value),
              )}
            </Field>
          ))}

          {featuredOptionalSettings.map(parameter => (
            <Field
              key={parameter.name}
              label={humanizeParameterLabel(parameter)}
              hint={humanizeParameterHint(parameter)}
            >
              {renderParameterControl(
                parameter,
                formState.parameterValues[parameter.name],
                styles.select,
                value => onParameterChange(parameter.name, value),
              )}
            </Field>
          ))}
        </div>

        {internalDefaultSettings.length > 0 && (
          <div className={styles.noteBox}>
            <Text weight="semibold">Handled automatically</Text>
            <Text className={styles.helper} block>
              This option already uses PyRIT&apos;s configured helper model and built-in template defaults, so you do not need to fill those in here.
            </Text>
          </div>
        )}

        {remainingOptionalSettings.length > 0 && (
          <details className={styles.details}>
            <summary className={styles.detailsSummary}>More option settings</summary>
            <div className={styles.fieldGrid}>
              {remainingOptionalSettings.map(parameter => (
                <Field
                  key={parameter.name}
                  label={humanizeParameterLabel(parameter)}
                  hint={humanizeParameterHint(parameter)}
                >
                  {renderParameterControl(
                    parameter,
                    formState.parameterValues[parameter.name],
                    styles.select,
                    value => onParameterChange(parameter.name, value),
                  )}
                </Field>
              ))}
            </div>
          </details>
        )}

        {visibleUnsupportedSettings.length > 0 && (
          <details className={styles.details}>
            <summary className={styles.detailsSummary}>Settings that still need extra setup</summary>
            <div className={styles.unsupportedList}>
              {visibleUnsupportedSettings.map(parameter => (
                <div key={parameter.name}>
                  <Text weight="semibold">{humanizeParameterLabel(parameter)}</Text>
                  <Text className={styles.helper} block>
                    This option is still visible here, but this setting needs a more custom input than the current form supports.
                  </Text>
                </div>
              ))}
            </div>
          </details>
        )}

        <div className={styles.actionRow}>
          <Button appearance="secondary" onClick={onClearOptionSettings}>
            Clear option settings
          </Button>
        </div>
      </div>
    </section>
  )
}
