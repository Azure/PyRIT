import {
  Badge,
  Button,
  Dropdown,
  Field,
  Option,
  Switch,
  Text,
  Textarea,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import type {
  BuilderConfigResponse,
  ConverterTypeMetadata,
  PromptBankPreset,
  PromptBuilderFormState,
  ReferenceImageResponse,
} from '../../types'
import {
  canRequestVariants,
  canUseAttackStarter,
  humanizeOptionName,
} from './builderUtils'

const useStyles = makeStyles({
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
  fieldGrid: {
    display: 'grid',
    gap: tokens.spacingVerticalM,
  },
  noteBox: {
    borderRadius: tokens.borderRadiusMedium,
    backgroundColor: tokens.colorNeutralBackground2,
    padding: tokens.spacingHorizontalM,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  actionRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalS,
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
})

interface AttackStarterPanelProps {
  option: ConverterTypeMetadata | null
  config: BuilderConfigResponse | null
  formState: PromptBuilderFormState
  selectedFamilyId: string
  selectedPreset: PromptBankPreset | null
  latestGeneratedImage: ReferenceImageResponse | null
  onFamilyChange: (value: string) => void
  onPresetChange: (presetId: string) => void
  onPresetFieldChange: (name: string, value: string) => void
  onApplyPreset: () => void
  onAvoidBlockedWordsChange: (value: boolean) => void
  onBlockedWordsTextChange: (value: string) => void
  onVariantCountChange: (value: number) => void
  onUseLatestGeneratedImage: () => void
}

export default function AttackStarterPanel({
  option,
  config,
  formState,
  selectedFamilyId,
  selectedPreset,
  latestGeneratedImage,
  onFamilyChange,
  onPresetChange,
  onPresetFieldChange,
  onApplyPreset,
  onAvoidBlockedWordsChange,
  onBlockedWordsTextChange,
  onVariantCountChange,
  onUseLatestGeneratedImage,
}: AttackStarterPanelProps) {
  const styles = useStyles()

  if (!option || !config) {
    return null
  }

  const supportsStarter = canUseAttackStarter(option)
  const supportsVariants = canRequestVariants(option, config)
  const presets = config.presets.filter(preset => preset.family_id === selectedFamilyId)
  const maxVariantCount = config.defaults.max_variant_count

  return (
    <section className={styles.panel}>
      <div>
        <Text as="h2" size={500} weight="semibold">Attack starter</Text>
        <Text className={styles.helper} block>
          Use a guided starter when you want speed, then keep editing manually to preserve precision.
        </Text>
      </div>

      <div className={styles.metaRow}>
        <Badge appearance="outline">{humanizeOptionName(option)}</Badge>
        <Badge appearance="outline">{supportsStarter ? 'Preset-ready' : 'Manual source flow'}</Badge>
        <Badge appearance="outline">{supportsVariants ? 'Multiple text versions supported' : 'Single output only'}</Badge>
      </div>

      {supportsStarter ? (
        <>
          <Field label="Attack family" hint="These are guided starting points from the video red-team report.">
            <Dropdown
              className={styles.select}
              aria-label="Attack family"
              selectedOptions={selectedFamilyId ? [selectedFamilyId] : []}
              value={config.families.find(family => family.family_id === selectedFamilyId)?.title || ''}
              onOptionSelect={(_, data) => onFamilyChange(data.optionValue || '')}
            >
              {config.families.map(family => (
                <Option key={family.family_id} value={family.family_id} text={family.title}>
                  {family.title}
                </Option>
              ))}
            </Dropdown>
          </Field>

          <Field label="Starter preset" hint={selectedPreset?.summary || 'Pick a preset, fill the blanks, then apply it into the prompt box.'}>
            <Dropdown
              className={styles.select}
              aria-label="Starter preset"
              selectedOptions={formState.selectedPresetId ? [formState.selectedPresetId] : []}
              value={selectedPreset?.title || ''}
              onOptionSelect={(_, data) => onPresetChange(data.optionValue || '')}
            >
              {presets.map(preset => (
                <Option key={preset.preset_id} value={preset.preset_id} text={preset.title}>
                  {preset.title}
                </Option>
              ))}
            </Dropdown>
          </Field>

          {selectedPreset && (
            <div className={styles.fieldGrid}>
              {selectedPreset.fields.map(field => (
                <Field
                  key={field.name}
                  label={`${field.label}${field.required ? ' *' : ''}`}
                  hint={field.placeholder || undefined}
                >
                  <Textarea
                    value={formState.presetValues[field.name] || ''}
                    onChange={(_, data) => onPresetFieldChange(field.name, data.value)}
                    placeholder={field.placeholder || field.label}
                    rows={2}
                  />
                </Field>
              ))}
              <div className={styles.actionRow}>
                <Button appearance="secondary" onClick={onApplyPreset}>
                  Apply starter to prompt
                </Button>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className={styles.noteBox}>
          <Text weight="semibold">Attack starters are text-first</Text>
          <Text className={styles.helper} block>
            This option starts from a file or media path, so the preset bank does not overwrite its source field.
          </Text>
        </div>
      )}

      <Field
        label="Avoid obviously blocked words"
        hint="This asks the helper model to rephrase common trigger words before the main converter runs."
      >
        <Switch
          checked={formState.avoidBlockedWords}
          onChange={(_, data) => onAvoidBlockedWordsChange(data.checked)}
          label={formState.avoidBlockedWords ? 'On' : 'Off'}
        />
      </Field>

      {formState.avoidBlockedWords && (
        <Field label="Blocked words list" hint="One word or phrase per line. You can adjust the defaults for this run.">
          <Textarea
            value={formState.blockedWordsText}
            onChange={(_, data) => onBlockedWordsTextChange(data.value)}
            rows={5}
          />
        </Field>
      )}

      <Field
        label="Versions"
        hint={
          supportsVariants
            ? 'Return the base output plus additional rewritten versions.'
            : 'This option only returns one output version.'
        }
      >
        <Dropdown
          className={styles.select}
          aria-label="Versions"
          selectedOptions={[String(formState.variantCount)]}
          value={String(formState.variantCount)}
          onOptionSelect={(_, data) => onVariantCountChange(Number(data.optionValue || '1'))}
          disabled={!supportsVariants}
        >
          {Array.from({ length: maxVariantCount }, (_, index) => String(index + 1)).map(value => (
            <Option key={value} value={value} text={value}>
              {value}
            </Option>
          ))}
        </Dropdown>
      </Field>

      {latestGeneratedImage && option.supported_input_types.includes('image_path') && (
        <div className={styles.noteBox}>
          <Text weight="semibold">Latest generated reference image</Text>
          <Text className={styles.helper} block>
            Reuse the most recent generated image as the input for this image-based option.
          </Text>
          <div className={styles.actionRow}>
            <Button appearance="secondary" onClick={onUseLatestGeneratedImage}>
              Use latest generated image
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
