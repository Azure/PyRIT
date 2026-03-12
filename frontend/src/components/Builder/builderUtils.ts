import type {
  ConverterParameterMetadata,
  ConverterTypeMetadata,
  PromptBuilderFormState,
} from '../../types'

const optionDescriptionOverrides: Record<string, string> = {
  AddImageTextConverter: 'Adds your text onto an image.',
  AddImageVideoConverter: 'Places an image into a video.',
  AddTextImageConverter: 'Adds text onto an existing image.',
  AsciiArtConverter: 'Turns text into ASCII art.',
  AsciiSmugglerConverter: 'Hides text inside unicode tags.',
  HumanInTheLoopConverter: 'Lets a person review each prompt before it is used.',
  JsonStringConverter: 'Turns text into JSON-safe text.',
  LLMGenericTextConverter: 'Uses another model to rewrite or transform text.',
  SelectiveTextConverter: 'Applies another option only to selected parts of the text.',
  WordDocConverter: 'Converts a text prompt into a Word document.',
}

const optionNameOverrides: Record<string, string> = {
  LLMGenericTextConverter: 'Text rewrite with another model',
  QRCodeConverter: 'QR code',
}

const longPromptFriendlyConverters = new Set([
  'LLMGenericTextConverter',
  'NoiseConverter',
  'PersuasionConverter',
  'RepeatTokenConverter',
  'ScientificTranslationConverter',
  'SuffixAppendConverter',
  'TenseConverter',
  'ToneConverter',
  'TranslationConverter',
  'VariationConverter',
])

const videoPromptFriendlyConverters = new Set([
  'LLMGenericTextConverter',
  'NegationTrapConverter',
  'NoiseConverter',
  'PersuasionConverter',
  'RepeatTokenConverter',
  'SuffixAppendConverter',
  'TenseConverter',
  'ToneConverter',
  'TranslationConverter',
  'VariationConverter',
])

const imageFriendlyConverterNames = new Set([
  'AddImageTextConverter',
  'AddImageVideoConverter',
  'AddTextImageConverter',
  'ImageCompressionConverter',
  'TransparencyAttackConverter',
])

const parameterLabelOverrides: Record<string, string> = {
  img_to_add: 'Image to add',
  encoding_func: 'Encoding style',
  converter_target: 'Model used for the rewrite',
  prompt_template: 'Template to use',
  word_selection_strategy: 'Which words to change',
  jailbreak_template: 'Jailbreak pattern',
  word_swap_ratio: 'How much to change',
  between_words: 'Only between words',
  caesar_offset: 'Shift amount',
  target_chars: 'Characters to change',
  accent: 'Accent style',
  suffix: 'Text to append',
  pattern: 'Text or pattern to match',
  replace: 'Replacement text',
  regex_flags: 'Regex options',
  action: 'Action',
  unicode_tags: 'Use unicode tags',
  percentage: 'How much to change',
  persuasion_technique: 'Persuasion technique',
  length_mode: 'Prompt length',
  converter: 'Wrapped option',
  benign_question_path: 'Benign image path',
  text_path: 'Text file path',
}

const parameterHintOverrides: Record<string, string> = {
  img_to_add: 'Path to the image file this option should use.',
  converter_target: 'The rewrite model or helper this option depends on.',
  persuasion_technique: 'Pick the persuasion style this option should use.',
  length_mode: 'Choose whether the rewrite should stay normal, become more detailed, or try to use most of a 3500-character prompt box.',
  word_selection_strategy: 'How this option decides which words to alter.',
  converter: 'The other option this wrapper should apply.',
}

export function humanizeOptionDescription(option: ConverterTypeMetadata) {
  const override = optionDescriptionOverrides[option.converter_type]
  if (override) {
    return override
  }

  const cleaned = option.description
    .replace(/`/g, '')
    .replace(/^Converter that /i, '')
    .replace(/^Base class for converters that /i, '')
    .replace(/\ban LLM\b/g, 'another model')
    .replace(/\ba LLM\b/g, 'another model')
    .replace(/\bLLM behavior\b/g, 'model behavior')
    .replace(/\bLLM\b/g, 'model')
    .replace(/^encodes text to /i, 'Encodes text as ')
    .replace(/^encodes prompts using /i, 'Encodes prompts using ')
    .replace(/\s+/g, ' ')
    .trim()

  if (!cleaned) {
    return 'Transforms the prompt in this way.'
  }

  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
}

export function humanizeOptionName(option: ConverterTypeMetadata) {
  return optionNameOverrides[option.converter_type] || option.display_name
}

export function humanizeParameterLabel(parameter: ConverterParameterMetadata) {
  const override = parameterLabelOverrides[parameter.name]
  if (override) {
    return override
  }

  return parameter.display_name
}

export function humanizeParameterHint(parameter: ConverterParameterMetadata) {
  const override = parameterHintOverrides[parameter.name]
  if (override) {
    return override
  }

  if (parameter.default_value) {
    return `Default: ${parameter.default_value}`
  }

  if (parameter.input_kind === 'select' && parameter.options && parameter.options.length > 0) {
    return `Choose one of: ${parameter.options.join(', ')}`
  }

  if (parameter.input_kind === 'list') {
    return 'Enter one item per line or separate items with commas.'
  }

  return `Expected value: ${parameter.type_label.toLowerCase()}`
}

export function humanizeDataType(value: string) {
  const label = value.toLowerCase()

  if (label === 'text') return 'Text'
  if (label === 'image_path') return 'Image'
  if (label === 'audio_path') return 'Audio'
  if (label === 'video_path') return 'Video'
  if (label === 'binary_path') return 'File'
  if (label === 'url') return 'URL'
  if (label === 'unknown') return 'Unknown'

  return value.replace(/_/g, ' ')
}

export function getPrimaryInputType(option: ConverterTypeMetadata | null) {
  return option?.supported_input_types[0] || 'text'
}

export function getPrimaryOutputType(option: ConverterTypeMetadata | null) {
  return option?.supported_output_types[0] || 'text'
}

export function getSourceFieldLabel(option: ConverterTypeMetadata | null) {
  const inputType = getPrimaryInputType(option)

  if (inputType === 'text') {
    return 'Text to transform'
  }

  if (inputType === 'url') {
    return 'URL to transform'
  }

  if (inputType === 'image_path') {
    return 'Reference image path or URL'
  }

  if (inputType === 'video_path') {
    return 'Source video path or URL'
  }

  return `${humanizeDataType(inputType)} path or source value`
}

export function getSourceFieldHint(option: ConverterTypeMetadata | null) {
  const inputTypes = option?.supported_input_types || []
  const outputTypes = option?.supported_output_types || []
  const inputLabel = inputTypes.map(humanizeDataType).join(', ') || 'Unknown'
  const outputLabel = outputTypes.map(humanizeDataType).join(', ') || 'Unknown'

  if ((inputTypes[0] || 'text') === 'text') {
    return `PyRIT usually starts with an existing prompt or source text, then applies this option. This one takes ${inputLabel.toLowerCase()} in and produces ${outputLabel.toLowerCase()} out.`
  }

  if ((inputTypes[0] || 'text') === 'image_path') {
    return `Use a saved image path or URL here. This option starts with ${inputLabel.toLowerCase()} and produces ${outputLabel.toLowerCase()} output.`
  }

  if ((inputTypes[0] || 'text') === 'video_path') {
    return `Use a saved video path or URL here. This option starts with ${inputLabel.toLowerCase()} and produces ${outputLabel.toLowerCase()} output.`
  }

  return `Enter the starting ${inputLabel.toLowerCase()} value PyRIT should use. This option produces ${outputLabel.toLowerCase()} output.`
}

export function getSourceFieldPlaceholder(option: ConverterTypeMetadata | null) {
  const inputType = getPrimaryInputType(option)

  if (inputType === 'text') {
    return 'Paste the starting prompt or source text you want this option to transform.'
  }

  if (inputType === 'url') {
    return 'https://example.com/path'
  }

  if (inputType === 'image_path') {
    return '/path/to/image.png or a stored image URL'
  }

  if (inputType === 'audio_path') {
    return '/path/to/audio.wav'
  }

  if (inputType === 'video_path') {
    return '/path/to/video.mp4 or a stored video URL'
  }

  if (inputType === 'binary_path') {
    return '/path/to/file.pdf'
  }

  return 'Enter the source value this option should use.'
}

export function formatOptionFlow(option: ConverterTypeMetadata) {
  const inputs = option.supported_input_types.map(humanizeDataType).join(', ') || 'Unknown'
  const outputs = option.supported_output_types.map(humanizeDataType).join(', ') || 'Unknown'
  return `${inputs} in -> ${outputs} out`
}

export function formatRequiredSetupLabel(option: ConverterTypeMetadata) {
  const requiredCount = option.parameters.filter(parameter => parameter.required).length
  if (requiredCount === 0) {
    return 'No required setup'
  }

  return `${requiredCount} required setting${requiredCount === 1 ? '' : 's'}`
}

export function isGoodForLongPrompts(option: ConverterTypeMetadata) {
  return longPromptFriendlyConverters.has(option.converter_type)
}

export function isVideoSpecificOption(option: ConverterTypeMetadata) {
  return option.supported_input_types.includes('video_path') || option.supported_output_types.includes('video_path')
}

export function worksWithImageUploads(option: ConverterTypeMetadata) {
  return (
    imageFriendlyConverterNames.has(option.converter_type) ||
    option.supported_input_types.includes('image_path')
  )
}

export function isGoodForVideoPromptTesting(option: ConverterTypeMetadata) {
  return videoPromptFriendlyConverters.has(option.converter_type)
}

export function isTextOnlyOption(option: ConverterTypeMetadata) {
  return (
    option.supported_input_types.length > 0 &&
    option.supported_output_types.length > 0 &&
    option.supported_input_types.every(type => type === 'text') &&
    option.supported_output_types.every(type => type === 'text')
  )
}

export function getVideoWorkflowGuidance(option: ConverterTypeMetadata) {
  if (isVideoSpecificOption(option)) {
    return {
      title: 'Direct video workflow',
      body: 'This option is most useful when you already have a video file, want to modify one, or want a video file as the result.',
    }
  }

  if (worksWithImageUploads(option)) {
    return {
      title: 'Helpful when image upload is part of the attack',
      body: 'This option works with image inputs or outputs, so it fits flows where the video model accepts a reference image or first frame.',
    }
  }

  if (isGoodForVideoPromptTesting(option)) {
    return {
      title: 'Good for video prompt attacks',
      body: 'This option changes the wording of the text prompt. That is often the main lever when you are testing a video generator.',
    }
  }

  if (isTextOnlyOption(option)) {
    return {
      title: 'Text-only option',
      body: 'This one only changes the prompt text. It does not change image or video files directly.',
    }
  }

  return {
    title: 'General option',
    body: 'This option is still available, but it is not especially tailored to video-generator testing.',
  }
}

export function buildPromptPreview(
  formState: PromptBuilderFormState,
) {
  return formState.sourceContent.trim() || '[Add the source content this option should transform]'
}

export function getMissingRequiredParams(
  option: ConverterTypeMetadata | null,
  parameterValues: PromptBuilderFormState['parameterValues'],
) {
  if (!option) {
    return []
  }

  return option.parameters.filter(param => {
    if (!param.required || param.input_kind === 'unsupported') {
      return false
    }

    const value = parameterValues[param.name]

    if (typeof value === 'boolean') {
      return false
    }

    return value === undefined || `${value}`.trim() === ''
  })
}

export function normalizeParameterValue(
  parameter: ConverterParameterMetadata,
  rawValue: string | number | boolean | undefined,
) {
  if (rawValue === undefined) {
    return undefined
  }

  if (parameter.input_kind === 'boolean') {
    return Boolean(rawValue)
  }

  if (parameter.input_kind === 'number') {
    if (`${rawValue}`.trim() === '') {
      return undefined
    }

    return Number(rawValue)
  }

  if (parameter.input_kind === 'list') {
    const text = `${rawValue}`.trim()
    if (!text) {
      return undefined
    }

    return text
      .split(/\n|,/)
      .map(item => item.trim())
      .filter(Boolean)
  }

  const text = `${rawValue}`.trim()
  return text ? text : undefined
}

export function buildPreviewParams(
  option: ConverterTypeMetadata | null,
  parameterValues: PromptBuilderFormState['parameterValues'],
) {
  if (!option) {
    return {}
  }

  return option.parameters.reduce<Record<string, unknown>>((params, parameter) => {
    const value = normalizeParameterValue(parameter, parameterValues[parameter.name])

    if (value !== undefined) {
      params[parameter.name] = value
    }

    return params
  }, {})
}

export function optionRequiresExtraSetup(option: ConverterTypeMetadata | null) {
  if (!option) {
    return false
  }

  return option.parameters.some(parameter => parameter.input_kind === 'unsupported')
}
