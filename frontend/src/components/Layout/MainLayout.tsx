import { ReactNode } from 'react'
import {
  makeStyles,
  tokens,
  Text,
} from '@fluentui/react-components'
import Navigation from '../Sidebar/Navigation'

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
  children: ReactNode
  onToggleTheme: () => void
  isDarkMode: boolean
  onNewChat: () => void
}

export default function MainLayout({ children, onToggleTheme, isDarkMode, onNewChat }: MainLayoutProps) {
  const styles = useStyles()

  return (
    <div className={styles.root}>
      <div className={styles.topBar}>
        <img 
          src="/roakey.png" 
          alt="Co-PyRIT Logo" 
          className={styles.logo}
        />
        <Text className={styles.title}>Co-PyRIT</Text>
        <Text className={styles.subtitle}>Python Risk Identification Tool</Text>
      </div>
      <div className={styles.contentArea}>
        <aside className={styles.sidebar}>
          <Navigation onToggleTheme={onToggleTheme} isDarkMode={isDarkMode} onNewChat={onNewChat} />
        </aside>
        <main className={styles.main}>{children}</main>
      </div>
    </div>
  )
}
