import { useState, useEffect } from 'react'
import {
  makeStyles,
  Button,
  tokens,
  Text,
  Select,
  Textarea,
  Divider,
  Input,
  Switch,
  Field,
  SpinButton,
} from '@fluentui/react-components'
import { DismissRegular, AddRegular } from '@fluentui/react-icons'
import { convertersApi, convertApi } from '../../services/api'
import { ConverterInfo, ConverterInstance } from '../../types'

const useStyles = makeStyles({
  panel: {
    height: '100%',
    width: '100%',
    backgroundColor: tokens.colorNeutralBackground1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    padding: tokens.spacingVerticalL,
    borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexShrink: 0,
  },
  title: {
    fontSize: tokens.fontSizeBase500,
    fontWeight: tokens.fontWeightSemibold,
  },
  content: {
    flex: 1,
    padding: tokens.spacingVerticalL,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalL,
    overflowY: 'auto',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
  },
  sectionLabel: {
    fontSize: tokens.fontSizeBase200,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground2,
  },
  converterItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorNeutralBackground3,
    borderRadius: tokens.borderRadiusMedium,
    position: 'relative',
  },
  converterHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
  },
  dragHandle: {
    cursor: 'grab',
    color: tokens.colorNeutralForeground3,
  },
  converterSelect: {
    flex: 1,
  },
  removeButton: {
    minWidth: '32px',
    width: '32px',
    height: '32px',
  },
  addButton: {
    alignSelf: 'flex-start',
  },
  footer: {
    padding: tokens.spacingVerticalL,
    borderTop: `1px solid ${tokens.colorNeutralStroke1}`,
    display: 'flex',
    gap: tokens.spacingHorizontalM,
    justifyContent: 'flex-end',
  },
  previewSection: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalS,
    marginTop: tokens.spacingVerticalM,
  },
  previewLabel: {
    fontSize: tokens.fontSizeBase200,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorNeutralForeground3,
  },
  previewContent: {
    padding: tokens.spacingVerticalM,
    backgroundColor: tokens.colorNeutralBackground4,
    borderRadius: tokens.borderRadiusMedium,
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground2,
    minHeight: '60px',
  },
})

interface ConverterDrawerProps {
  isOpen: boolean
  onClose: () => void
  onApply: (text: string, converters: ConverterInstance[], converterIdentifiers: Array<Record<string, string>>) => void
  initialText?: string
}

export default function ConverterDrawer({ isOpen, onClose, onApply, initialText = '' }: ConverterDrawerProps) {
  const styles = useStyles()
  const [text, setText] = useState(initialText)
  const [converters, setConverters] = useState<ConverterInstance[]>([])
  const [availableConverters, setAvailableConverters] = useState<ConverterInfo[]>([])
  const [previewText, setPreviewText] = useState('')
  const [generatingPreview, setGeneratingPreview] = useState(false)
  const [converterIdentifiers, setConverterIdentifiers] = useState<Array<Record<string, string>>>([])

  useEffect(() => {
    setText(initialText)
  }, [initialText])

  useEffect(() => {
    if (isOpen) {
      loadConverters()
    }
  }, [isOpen])

  const loadConverters = async () => {
    try {
      const data = await convertersApi.getConverters()
      setAvailableConverters(data)
    } catch (error) {
      console.error('Failed to load converters:', error)
    }
  }

  // Auto-preview for non-LLM converters
  useEffect(() => {
    if (!text || converters.length === 0) {
      setPreviewText('')
      return
    }

    // Check if all converters are non-LLM
    const allNonLLM = converters.every(conv => {
      const info = availableConverters.find(c => c.class_name === conv.class_name)
      return info && !info.uses_llm
    })

    if (allNonLLM) {
      // Auto-generate preview for non-LLM converters via API
      generatePreview()
    } else {
      // Clear preview if there are LLM converters (they need manual generation)
      setPreviewText('')
    }
  }, [text, converters, availableConverters])

  const generatePreview = async () => {
    if (!text || converters.length === 0) {
      setPreviewText('')
      return
    }

    setGeneratingPreview(true)
    try {
      const converterConfigs = converters.map(c => ({
        class_name: c.class_name,
        config: c.config
      }))
      
      const result = await convertApi.convert(text, converterConfigs)
      setPreviewText(result.converted_text)
      setConverterIdentifiers(result.converter_identifiers || [])
    } catch (error) {
      console.error('Preview generation failed:', error)
      setPreviewText('[Preview generation failed: ' + (error as any).response?.data?.detail || 'Unknown error' + ']')
    } finally {
      setGeneratingPreview(false)
    }
  }

  const handleAddConverter = () => {
    if (availableConverters.length === 0) return
    
    const firstConverter = availableConverters[0]
    const defaultConfig: Record<string, any> = {}
    
    // Initialize config with default values
    firstConverter.parameters.forEach(param => {
      if (param.default !== undefined && param.default !== null) {
        defaultConfig[param.name] = param.default
      } else if (param.type === 'bool') {
        defaultConfig[param.name] = false
      } else if (param.type === 'int' || param.type === 'float') {
        defaultConfig[param.name] = 0
      } else if (param.type === 'str') {
        defaultConfig[param.name] = ''
      } else if (param.type === 'enum' && param.enum_values && param.enum_values.length > 0) {
        defaultConfig[param.name] = param.enum_values[0]
      }
    })
    
    const newConverter: ConverterInstance = {
      id: `converter-${Date.now()}`,
      class_name: firstConverter.class_name,
      config: defaultConfig,
    }
    setConverters([...converters, newConverter])
  }

  const handleRemoveConverter = (id: string) => {
    setConverters(converters.filter((c) => c.id !== id))
  }

  const handleConverterTypeChange = (id: string, newClassName: string) => {
    const converterInfo = availableConverters.find(c => c.class_name === newClassName)
    if (!converterInfo) return
    
    const defaultConfig: Record<string, any> = {}
    converterInfo.parameters.forEach(param => {
      if (param.default !== undefined && param.default !== null) {
        defaultConfig[param.name] = param.default
      } else if (param.type === 'bool') {
        defaultConfig[param.name] = false
      } else if (param.type === 'int' || param.type === 'float') {
        defaultConfig[param.name] = 0
      } else if (param.type === 'str') {
        defaultConfig[param.name] = ''
      } else if (param.type === 'enum' && param.enum_values && param.enum_values.length > 0) {
        defaultConfig[param.name] = param.enum_values[0]
      }
    })
    
    setConverters(
      converters.map((c) => 
        c.id === id ? { ...c, class_name: newClassName, config: defaultConfig } : c
      )
    )
  }

  const handleConfigChange = (converterId: string, paramName: string, value: any) => {
    setConverters(
      converters.map((c) => 
        c.id === converterId 
          ? { ...c, config: { ...c.config, [paramName]: value } }
          : c
      )
    )
  }

  const handleDiscard = () => {
    setText(initialText)
    setConverters([])
    setPreviewText('')
    setConverterIdentifiers([])
    onClose()
  }

  const handleApply = () => {
    // Apply the preview text if available, otherwise use original text
    const textToApply = previewText || text
    onApply(textToApply, converters, converterIdentifiers)
    // Don't close the panel - keep it open with converters configured
  }

  if (!isOpen) return null

  return (
    <div className={styles.panel}>
      <div className={styles.header}>
        <Text className={styles.title}>Prompt Converter</Text>
        <Button
          appearance="subtle"
          icon={<DismissRegular />}
          onClick={handleDiscard}
          title="Close"
        />
      </div>

      <div className={styles.content}>
        <div className={styles.section}>
          <Text className={styles.sectionLabel}>Input</Text>
            <Textarea
              placeholder="Type something here"
              value={text}
              onChange={(_, data) => setText(data.value)}
              resize="vertical"
              rows={4}
            />
          </div>

          <Divider />

          <div className={styles.section}>
            <Text className={styles.sectionLabel}>Converters</Text>
            {converters.map((converter) => {
              const converterInfo = availableConverters.find(c => c.class_name === converter.class_name)
              
              return (
                <div key={converter.id} className={styles.converterItem}>
                  <div className={styles.converterHeader}>
                    <Text className={styles.dragHandle}>::</Text>
                    <Select
                      className={styles.converterSelect}
                      value={converter.class_name}
                      onChange={(_, data) => handleConverterTypeChange(converter.id, data.value)}
                    >
                      {availableConverters.map((info) => (
                        <option key={info.class_name} value={info.class_name}>
                          {info.name}
                        </option>
                      ))}
                    </Select>
                    <Button
                      className={styles.removeButton}
                      appearance="subtle"
                      icon={<DismissRegular />}
                      onClick={() => handleRemoveConverter(converter.id)}
                      title="Remove converter"
                    />
                  </div>
                  
                  {/* Render parameters dynamically */}
                  {converterInfo && converterInfo.parameters.length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalS, marginTop: tokens.spacingVerticalXS }}>
                      {converterInfo.parameters.map((param) => {
                        const value = converter.config[param.name]
                        const label = (
                          <>
                            {param.name} {param.required ? <span style={{ color: 'red' }}>*</span> : <span style={{ color: tokens.colorNeutralForeground3, fontSize: '0.9em' }}>(optional)</span>}
                          </>
                        )
                        
                        if (param.type === 'bool') {
                          return (
                            <Field key={param.name} label={label}>
                              <Switch
                                checked={value || false}
                                onChange={(_, data) => handleConfigChange(converter.id, param.name, data.checked)}
                              />
                            </Field>
                          )
                        } else if (param.type === 'int') {
                          return (
                            <Field key={param.name} label={label}>
                              <SpinButton
                                value={value || 0}
                                min={0}
                                step={1}
                                onChange={(_, data) => {
                                  if (data.value !== undefined && data.value !== null) {
                                    handleConfigChange(converter.id, param.name, data.value)
                                  }
                                }}
                              />
                            </Field>
                          )
                        } else if (param.type === 'float') {
                          return (
                            <Field key={param.name} label={label}>
                              <Input
                                type="number"
                                value={value?.toString() || '0'}
                                onChange={(_, data) => handleConfigChange(converter.id, param.name, parseFloat(data.value) || 0)}
                              />
                            </Field>
                          )
                        } else if (param.type === 'enum' && param.enum_values) {
                          return (
                            <Field key={param.name} label={label}>
                              <Select
                                value={value || param.enum_values[0]}
                                onChange={(_, data) => handleConfigChange(converter.id, param.name, data.value)}
                              >
                                {param.enum_values.map((option) => (
                                  <option key={option} value={option}>
                                    {option}
                                  </option>
                                ))}
                              </Select>
                            </Field>
                          )
                        } else if (param.type === 'str') {
                          return (
                            <Field key={param.name} label={label}>
                              <Input
                                value={value || ''}
                                onChange={(_, data) => handleConfigChange(converter.id, param.name, data.value)}
                                placeholder={param.description || `Enter ${param.name}`}
                              />
                            </Field>
                          )
                        }
                        return null
                      })}
                    </div>
                  )}
                </div>
              )
            })}
            <Button
              className={styles.addButton}
              appearance="outline"
              icon={<AddRegular />}
              onClick={handleAddConverter}
              disabled={availableConverters.length === 0}
            >
              Add
            </Button>
          </div>

          {converters.length > 0 && (
            <div className={styles.previewSection}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Text className={styles.previewLabel}>Preview content</Text>
                {converters.some(conv => {
                  const info = availableConverters.find(c => c.class_name === conv.class_name)
                  return info && info.uses_llm
                }) && (
                  <Button
                    appearance="primary"
                    size="small"
                    disabled={!text || generatingPreview}
                    onClick={generatePreview}
                  >
                    {generatingPreview ? 'Generating...' : 'Generate Preview'}
                  </Button>
                )}
              </div>
              <div className={styles.previewContent}>
                {generatingPreview ? (
                  <Text>Generating preview...</Text>
                ) : previewText ? (
                  <Text>{previewText}</Text>
                ) : text ? (
                  <Text style={{ color: tokens.colorNeutralForeground3 }}>
                    {converters.some(conv => {
                      const info = availableConverters.find(c => c.class_name === conv.class_name)
                      return info && info.uses_llm
                    })
                      ? 'Click "Generate Preview" to see the converted text'
                      : 'Preview will appear here'}
                  </Text>
                ) : (
                  <Text style={{ color: tokens.colorNeutralForeground3 }}>Enter text to preview conversion</Text>
                )}
              </div>
            </div>
          )}

          <div className={styles.footer}>
            <Button appearance="secondary" onClick={handleDiscard}>
              Discard
            </Button>
            <Button appearance="primary" onClick={handleApply}>
              Send
            </Button>
          </div>
        </div>
    </div>
  )
}
