import { useMemo } from 'react'
import {
  Badge,
  Button,
  Input,
  Text,
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import type { ConverterTypeMetadata } from '../../types'
import {
  formatOptionFlow,
  formatRequiredSetupLabel,
  getVideoWorkflowGuidance,
  humanizeOptionDescription,
  humanizeOptionName,
  isGoodForLongPrompts,
  isGoodForVideoPromptTesting,
  isVideoSpecificOption,
  worksWithImageUploads,
} from './builderUtils'

type OptionSectionKey = 'text' | 'image' | 'audio' | 'video' | 'file' | 'other'

interface OptionSection {
  key: OptionSectionKey
  title: string
  description: string
}

const optionSections: OptionSection[] = [
  {
    key: 'text',
    title: 'Text-based options',
    description: 'Prompt rewrites, encodings, obfuscation, and wording changes.',
  },
  {
    key: 'image',
    title: 'Image options',
    description: 'Options that create, change, or rely on image content.',
  },
  {
    key: 'audio',
    title: 'Audio options',
    description: 'Options that turn prompts into audio or alter audio files.',
  },
  {
    key: 'video',
    title: 'Video options',
    description: 'Options that work with video inputs or outputs.',
  },
  {
    key: 'file',
    title: 'File and document options',
    description: 'Options that generate or modify files such as PDFs or Word documents.',
  },
  {
    key: 'other',
    title: 'Other options',
    description: 'Anything that does not fit the main buckets cleanly.',
  },
]

function getOptionSection(option: ConverterTypeMetadata): OptionSectionKey {
  const modalities = [...option.supported_input_types, ...option.supported_output_types]
    .join(' ')
    .toLowerCase()

  if (modalities.includes('audio')) {
    return 'audio'
  }

  if (modalities.includes('video')) {
    return 'video'
  }

  if (modalities.includes('image')) {
    return 'image'
  }

  if (modalities.includes('binary')) {
    return 'file'
  }

  if (modalities.includes('text') || modalities.includes('unknown')) {
    return 'text'
  }

  return 'other'
}

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    height: '100%',
    minHeight: 0,
    overflowY: 'auto',
    overflowX: 'hidden',
  },
  list: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
  },
  sectionHeader: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXXS,
    position: 'sticky',
    top: 0,
    zIndex: 1,
    backgroundColor: tokens.colorNeutralBackground2,
    paddingTop: tokens.spacingVerticalXS,
  },
  sectionCards: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
  },
  card: {
    fontFamily: 'inherit',
    textAlign: 'left',
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    borderRadius: tokens.borderRadiusLarge,
    padding: tokens.spacingHorizontalM,
    backgroundColor: tokens.colorNeutralBackground1,
    cursor: 'pointer',
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  selectedCard: {
    boxShadow: `inset 0 0 0 1px ${tokens.colorBrandStroke1}`,
    backgroundColor: tokens.colorBrandBackground2,
  },
  matchingCard: {
    boxShadow: `inset 0 0 0 1px ${tokens.colorPaletteYellowBorderActive}`,
  },
  metaRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalXS,
  },
  helper: {
    color: tokens.colorNeutralForeground3,
  },
  jumpRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalXS,
  },
  searchHelper: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXXS,
  },
  matchesRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalXS,
  },
  focusArea: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
    padding: tokens.spacingHorizontalM,
    borderRadius: tokens.borderRadiusLarge,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    backgroundColor: tokens.colorNeutralBackground1,
  },
  focusGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXXS,
  },
  shortcutRow: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalXS,
  },
})

interface AttackOptionListProps {
  options: ConverterTypeMetadata[]
  searchValue: string
  selectedConverterType: string | null
  onSearchChange: (value: string) => void
  onSelect: (option: ConverterTypeMetadata) => void
}

export default function AttackOptionList({
  options,
  searchValue,
  selectedConverterType,
  onSearchChange,
  onSelect,
}: AttackOptionListProps) {
  const styles = useStyles()
  const normalizedSearch = searchValue.trim().toLowerCase()

  const sectionedOptions = useMemo(() => {
    return optionSections
      .map(section => ({
        ...section,
        items: options.filter(option => getOptionSection(option) === section.key),
      }))
      .filter(section => section.items.length > 0)
  }, [options])

  const matchingOptions = useMemo(() => {
    if (!normalizedSearch) {
      return []
    }

    return options.filter(option =>
      [option.display_name, option.converter_type, option.description]
        .join(' ')
        .toLowerCase()
        .includes(normalizedSearch),
    )
  }, [normalizedSearch, options])

  const videoStartHereGroups = useMemo(
    () => [
      {
        key: 'video-first',
        title: 'Direct video options',
        description: 'Best when you already have a video file or want a video file as the result.',
        items: options.filter(isVideoSpecificOption),
      },
      {
        key: 'image-upload',
        title: 'Useful when image upload is part of the attack',
        description: 'Good for reference-image and first-frame style workflows.',
        items: options.filter(option => !isVideoSpecificOption(option) && worksWithImageUploads(option)),
      },
      {
        key: 'video-prompt',
        title: 'Prompt rewrites worth trying against video models',
        description: 'These change the wording that drives the video generator.',
        items: options.filter(option => isGoodForVideoPromptTesting(option)),
      },
    ].filter(group => group.items.length > 0),
    [options],
  )

  return (
    <section className={styles.root}>
      <div>
        <Text as="h2" size={500} weight="semibold">Attack options</Text>
        <Text className={styles.helper} block>
          All options stay visible below. Browse by section first, then use search only if you want to jump to something specific.
        </Text>
      </div>

      <div className={styles.focusArea}>
        <Text weight="semibold">Start here for video generators</Text>
        <Text className={styles.helper} size={200}>
          These shortcuts do not hide anything below. They just bring the most relevant paths to the top when you are testing a video model and can supply images.
        </Text>

        {videoStartHereGroups.map(group => (
          <div key={group.key} className={styles.focusGroup}>
            <Text weight="semibold" size={200}>{group.title}</Text>
            <Text className={styles.helper} size={200}>{group.description}</Text>
            <div className={styles.shortcutRow}>
              {group.items.map(option => (
                <Button
                  key={`${group.key}-${option.converter_type}`}
                  appearance="secondary"
                  size="small"
                  onClick={() => {
                    document.getElementById(`option-card-${option.converter_type}`)?.scrollIntoView({
                      behavior: 'smooth',
                      block: 'center',
                    })
                    onSelect(option)
                  }}
                >
                  {humanizeOptionName(option)}
                </Button>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div className={styles.jumpRow}>
        {sectionedOptions.map(section => (
          <Button
            key={section.key}
            appearance="secondary"
            size="small"
            onClick={() => {
              document.getElementById(`option-section-${section.key}`)?.scrollIntoView({
                behavior: 'smooth',
                block: 'start',
              })
            }}
          >
            {section.title} ({section.items.length})
          </Button>
        ))}
      </div>

      <Input
        value={searchValue}
        onChange={(_, data) => onSearchChange(data.value)}
        placeholder="Jump to a name or keyword (optional)"
        aria-label="Search attack options"
      />

      <div className={styles.searchHelper}>
        <Text className={styles.helper} block>
          {options.length} option{options.length === 1 ? '' : 's'} total
        </Text>
        {normalizedSearch ? (
          <>
            <Text className={styles.helper} block>
              {matchingOptions.length} match{matchingOptions.length === 1 ? '' : 'es'} highlighted below.
            </Text>
            {matchingOptions.length > 0 && (
              <div className={styles.matchesRow}>
                {matchingOptions.slice(0, 8).map(option => (
                  <Button
                    key={option.converter_type}
                    appearance="subtle"
                    size="small"
                    onClick={() => {
                      document.getElementById(`option-card-${option.converter_type}`)?.scrollIntoView({
                        behavior: 'smooth',
                        block: 'center',
                      })
                      onSelect(option)
                    }}
                  >
                    {humanizeOptionName(option)}
                  </Button>
                ))}
              </div>
            )}
          </>
        ) : null}
      </div>

      <div className={styles.list}>
        {sectionedOptions.map(section => (
          <div
            key={section.key}
            id={`option-section-${section.key}`}
            className={styles.section}
          >
            <div className={styles.sectionHeader}>
              <Text weight="semibold">{section.title}</Text>
              <Text className={styles.helper} size={200}>{section.description}</Text>
            </div>

            <div className={styles.sectionCards}>
              {section.items.map(option => {
                const unsupportedCount = option.parameters.filter(
                  parameter => parameter.input_kind === 'unsupported',
                ).length
                const isSelected = option.converter_type === selectedConverterType
                const isMatching =
                  normalizedSearch.length > 0 &&
                  [option.display_name, option.converter_type, option.description]
                    .join(' ')
                    .toLowerCase()
                    .includes(normalizedSearch)

                return (
                  <button
                    key={option.converter_type}
                    id={`option-card-${option.converter_type}`}
                    type="button"
                    className={`${styles.card} ${isSelected ? styles.selectedCard : ''} ${isMatching ? styles.matchingCard : ''}`}
                    onClick={() => onSelect(option)}
                  >
                    <Text weight="semibold">{humanizeOptionName(option)}</Text>
                    <Text size={200}>{humanizeOptionDescription(option)}</Text>
                    <div className={styles.metaRow}>
                      <Badge appearance="outline">{formatOptionFlow(option)}</Badge>
                      <Badge appearance="outline">{formatRequiredSetupLabel(option)}</Badge>
                      <Badge appearance="outline">{getVideoWorkflowGuidance(option).title}</Badge>
                      {isGoodForLongPrompts(option) && (
                        <Badge appearance="outline">Good for long prompts</Badge>
                      )}
                      {unsupportedCount > 0 && (
                        <Badge appearance="tint">{unsupportedCount} needs extra setup</Badge>
                      )}
                      {isMatching && (
                        <Badge appearance="filled">Match</Badge>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}
