import {
  makeStyles,
  tokens,
  Text,
  Tooltip,
} from '@fluentui/react-components'
import { useEffect, useState } from 'react'
import Navigation from '../Sidebar/Navigation'
import { apiClient } from '../../services/api'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
  },
  topBar: {
    height: '60px',
    backgroundColor: tokens.colorNeutralBackground3,
    borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
    display: 'flex',
    alignItems: 'center',
    padding: `0 ${tokens.spacingHorizontalL}`,
    gap: tokens.spacingHorizontalM,
  },
  logo: {
    width: '40px',
    height: '40px',
    cursor: 'help',
  },
  title: {
    fontSize: tokens.fontSizeHero700,
    fontWeight: tokens.fontWeightSemibold,
    color: tokens.colorBrandForeground1,
  },
  subtitle: {
    fontSize: tokens.fontSizeBase200,
    color: tokens.colorNeutralForeground3,
    marginLeft: tokens.spacingHorizontalXS,
  },
  contentArea: {
    display: 'flex',
    flex: 1,
    overflow: 'hidden',
  },
  sidebar: {
    width: '60px',
    backgroundColor: tokens.colorNeutralBackground3,
    borderRight: `1px solid ${tokens.colorNeutralStroke1}`,
    display: 'flex',
    flexDirection: 'column',
  },
  main: {
    flex: 1,
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
})

interface MainLayoutProps {
  children: React.ReactNode
  onToggleTheme: () => void
  isDarkMode: boolean
  onReturnToChat: () => void
  onShowHistory: () => void
  currentView: 'chat' | 'history'
}

export default function MainLayout({ 
  children, 
  onToggleTheme, 
  isDarkMode, 
  onReturnToChat, 
  onShowHistory, 
  currentView 
}: MainLayoutProps) {
  const styles = useStyles()
  const [version, setVersion] = useState<string>('Loading...')

  useEffect(() => {
    // Fetch version information
    apiClient.get<{version: string, display?: string}>('/api/version')
      .then((response) => {
        setVersion(response.data.display || response.data.version)
      })
      .catch(() => {
        setVersion('Unknown')
      })
  }, [])

  return (
    <div className={styles.root}>
      <div className={styles.topBar}>
        <Tooltip content={`PyRIT Version: ${version}`} relationship="label">
          <img 
            src="/roakey.png" 
            alt="Co-PyRIT Logo" 
            className={styles.logo}
          />
        </Tooltip>
        <Text className={styles.title}>Co-PyRIT</Text>
        <Text className={styles.subtitle}>Python Risk Identification Tool</Text>
      </div>
      <div className={styles.contentArea}>
        <aside className={styles.sidebar}>
          <Navigation 
            onToggleTheme={onToggleTheme} 
            isDarkMode={isDarkMode} 
            onReturnToChat={onReturnToChat}
            onShowHistory={onShowHistory}
            currentView={currentView}
          />
        </aside>
        <main className={styles.main}>{children}</main>
      </div>
    </div>
  )
}
