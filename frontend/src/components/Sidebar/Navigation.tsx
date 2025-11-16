import { useState, useEffect } from 'react'
import {
  makeStyles,
  Text,
  Button,
  tokens,
  Spinner,
} from '@fluentui/react-components'
import {
  ChatRegular,
  TargetArrowRegular,
  HistoryRegular,
  SettingsRegular,
} from '@fluentui/react-icons'
import { targetsApi } from '../../services/api'
import { TargetInfo } from '../../types'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    padding: tokens.spacingVerticalL,
  },
  header: {
    marginBottom: tokens.spacingVerticalXL,
  },
  title: {
    fontSize: tokens.fontSizeHero700,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorBrandForeground1,
  },
  subtitle: {
    fontSize: tokens.fontSizeBase300,
    color: tokens.colorNeutralForeground3,
    marginTop: tokens.spacingVerticalXS,
  },
  navSection: {
    marginBottom: tokens.spacingVerticalXXL,
  },
  navButton: {
    width: '100%',
    justifyContent: 'flex-start',
    marginBottom: tokens.spacingVerticalS,
  },
  targetsList: {
    flex: 1,
    overflowY: 'auto',
    marginTop: tokens.spacingVerticalM,
  },
  targetItem: {
    width: '100%',
    justifyContent: 'flex-start',
    marginBottom: tokens.spacingVerticalXS,
    padding: tokens.spacingHorizontalS,
  },
  loadingContainer: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    padding: tokens.spacingVerticalXL,
  },
})

export default function Navigation() {
  const styles = useStyles()
  const [targets, setTargets] = useState<TargetInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedTarget, setSelectedTarget] = useState<string | null>(null)

  useEffect(() => {
    loadTargets()
  }, [])

  const loadTargets = async () => {
    try {
      const data = await targetsApi.listTargets()
      setTargets(data)
    } catch (error) {
      console.error('Failed to load targets:', error)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <Text className={styles.title}>PyRIT</Text>
        <Text className={styles.subtitle}>AI Red Team Tool</Text>
      </div>

      <div className={styles.navSection}>
        <Button
          className={styles.navButton}
          appearance="subtle"
          icon={<ChatRegular />}
        >
          New Chat
        </Button>
        <Button
          className={styles.navButton}
          appearance="subtle"
          icon={<HistoryRegular />}
        >
          History
        </Button>
        <Button
          className={styles.navButton}
          appearance="subtle"
          icon={<SettingsRegular />}
        >
          Settings
        </Button>
      </div>

      <div className={styles.navSection}>
        <Text weight="semibold" block style={{ marginBottom: tokens.spacingVerticalS }}>
          <TargetArrowRegular style={{ marginRight: tokens.spacingHorizontalXS }} />
          Targets
        </Text>
        
        {loading ? (
          <div className={styles.loadingContainer}>
            <Spinner size="small" />
          </div>
        ) : (
          <div className={styles.targetsList}>
            {targets.map((target) => (
              <Button
                key={target.id}
                className={styles.targetItem}
                appearance={selectedTarget === target.id ? 'primary' : 'subtle'}
                onClick={() => setSelectedTarget(target.id)}
              >
                <div style={{ textAlign: 'left', width: '100%' }}>
                  <Text block weight="semibold">{target.name}</Text>
                  <Text block size={200} style={{ color: tokens.colorNeutralForeground3 }}>
                    {target.type}
                  </Text>
                </div>
              </Button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
