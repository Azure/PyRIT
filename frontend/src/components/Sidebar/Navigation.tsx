import { useState, useEffect } from 'react'
import {
  makeStyles,
  Text,
  Button,
  tokens,
  Dialog,
  DialogTrigger,
  DialogSurface,
  DialogTitle,
  DialogBody,
  DialogActions,
  DialogContent,
  Field,
  Select,
} from '@fluentui/react-components'
import {
  ChatRegular,
  SettingsRegular,
  WeatherMoonRegular,
  WeatherSunnyRegular,
} from '@fluentui/react-icons'
import { configApi } from '../../services/api'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    padding: tokens.spacingVerticalM,
    alignItems: 'center',
    gap: tokens.spacingVerticalM,
  },
  iconButton: {
    width: '44px',
    height: '44px',
    minWidth: '44px',
    padding: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  spacer: {
    flex: 1,
  },
})

interface NavigationProps {
  onToggleTheme: () => void
  isDarkMode: boolean
}

export default function Navigation({ onToggleTheme, isDarkMode }: NavigationProps) {
  const styles = useStyles()
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [targetConfig, setTargetConfig] = useState({
    targetType: '',
    apiKeyVar: '',
    endpointVar: '',
    modelVar: '',
  })
  const [keyVars, setKeyVars] = useState<string[]>([])
  const [endpointVars, setEndpointVars] = useState<string[]>([])
  const [modelVars, setModelVars] = useState<string[]>([])
  const [availableTargetTypes, setAvailableTargetTypes] = useState<any[]>([])
  const [endpointValue, setEndpointValue] = useState('')
  const [modelValue, setModelValue] = useState('')

  useEffect(() => {
    loadEnvVars()
    loadTargetTypes()
  }, [])

  const loadEnvVars = async () => {
    try {
      const data = await configApi.getEnvVars()
      setKeyVars(data.keys || [])
      setEndpointVars(data.endpoints || [])
      setModelVars(data.models || [])
    } catch (error) {
      console.error('Failed to load env vars:', error)
    }
  }

  const loadTargetTypes = async () => {
    try {
      const data = await configApi.getTargetTypes()
      setAvailableTargetTypes(data)
    } catch (error) {
      console.error('Failed to load target types:', error)
    }
  }

  const loadEnvVarValue = async (varName: string, field: 'endpoint' | 'model') => {
    try {
      const data = await configApi.getEnvVarValue(varName)
      if (field === 'endpoint') {
        setEndpointValue(data.value || '')
      } else {
        setModelValue(data.value || '')
      }
    } catch (error) {
      console.error('Failed to load env var value:', error)
    }
  }

  return (
    <div className={styles.root}>
      <Button
        className={styles.iconButton}
        appearance="subtle"
        icon={<ChatRegular />}
        title="Chat"
        disabled
      />
      
      <Dialog open={settingsOpen} onOpenChange={(_, data: any) => setSettingsOpen(data.open)}>
        <DialogTrigger disableButtonEnhancement>
          <Button
            className={styles.iconButton}
            appearance="subtle"
            icon={<SettingsRegular />}
            title="Target Settings"
          />
        </DialogTrigger>
        <DialogSurface>
          <DialogBody>
            <DialogTitle>Target Configuration</DialogTitle>
            <DialogContent>
              <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalM }}>
                <Field label="Target Type">
                  <Select
                    value={targetConfig.targetType}
                    onChange={(_, data) => {
                      const selectedType = availableTargetTypes.find(t => t.id === data.value)
                      const newConfig = {
                        targetType: data.value,
                        apiKeyVar: selectedType?.default_env_vars?.api_key || '',
                        endpointVar: selectedType?.default_env_vars?.endpoint || '',
                        modelVar: selectedType?.default_env_vars?.model || '',
                      }
                      setTargetConfig(newConfig)
                      // Clear old values
                      setEndpointValue('')
                      setModelValue('')
                      // Load values for endpoint and model if they're set
                      if (newConfig.endpointVar) {
                        loadEnvVarValue(newConfig.endpointVar, 'endpoint')
                      }
                      if (newConfig.modelVar) {
                        loadEnvVarValue(newConfig.modelVar, 'model')
                      }
                    }}
                  >
                    <option value="">Select target type...</option>
                    {availableTargetTypes.map((type) => (
                      <option key={type.id} value={type.id}>
                        {type.name}
                      </option>
                    ))}
                  </Select>
                </Field>
                {targetConfig.targetType && (
                  <>
                    <Field label="API Key Environment Variable">
                      <Select
                        value={targetConfig.apiKeyVar}
                        onChange={(_, data) => setTargetConfig({ ...targetConfig, apiKeyVar: data.value })}
                      >
                        <option value="">Select API key var...</option>
                          {keyVars.map((varName) => (
                          <option key={varName} value={varName}>
                            {varName}
                          </option>
                        ))}
                      </Select>
                    </Field>
                    <Field label="Endpoint Environment Variable">
                      <Select
                        value={targetConfig.endpointVar}
                        onChange={(_, data) => {
                          const newVar = data.value
                          setTargetConfig({ ...targetConfig, endpointVar: newVar })
                          if (newVar) loadEnvVarValue(newVar, 'endpoint')
                        }}
                      >
                        <option value="">Select endpoint var...</option>
                          {endpointVars.map((varName) => (
                            <option key={varName} value={varName}>
                              {varName}
                            </option>
                          ))}
                      </Select>
                      {endpointValue && (
                        <Text size={200} style={{ marginTop: tokens.spacingVerticalXS, color: tokens.colorNeutralForeground3 }}>
                          Value: {endpointValue}
                        </Text>
                      )}
                    </Field>
                    <Field label="Model Environment Variable">
                      <Select
                        value={targetConfig.modelVar}
                        onChange={(_, data) => {
                          const newVar = data.value
                          setTargetConfig({ ...targetConfig, modelVar: newVar })
                          if (newVar) loadEnvVarValue(newVar, 'model')
                        }}
                      >
                        <option value="">Select model var...</option>
                          {modelVars.map((varName) => (
                            <option key={varName} value={varName}>
                              {varName}
                            </option>
                          ))}
                      </Select>
                      {modelValue && (
                        <Text size={200} style={{ marginTop: tokens.spacingVerticalXS, color: tokens.colorNeutralForeground3 }}>
                          Value: {modelValue}
                        </Text>
                      )}
                    </Field>
                  </>
                )}
              </div>
            </DialogContent>
            <DialogActions>
              <DialogTrigger disableButtonEnhancement>
                <Button appearance="secondary">Cancel</Button>
              </DialogTrigger>
              <Button appearance="primary" onClick={() => {
                // Just store locally - will pass to server when making requests
                console.log('Target configured locally:', targetConfig)
                setSettingsOpen(false)
              }}>
                Save
              </Button>
            </DialogActions>
          </DialogBody>
        </DialogSurface>
      </Dialog>

      <div className={styles.spacer} />

      <Button
        className={styles.iconButton}
        appearance="subtle"
        icon={isDarkMode ? <WeatherSunnyRegular /> : <WeatherMoonRegular />}
        onClick={onToggleTheme}
        title={isDarkMode ? 'Light Mode' : 'Dark Mode'}
      />
    </div>
  )
}
