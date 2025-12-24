import {
  makeStyles,
  tokens,
  Text,
  Tooltip,
  Button,
} from '@fluentui/react-components'
import { WeatherMoonRegular, WeatherSunnyRegular } from '@fluentui/react-icons'

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
    justifyContent: 'space-between',
    padding: `0 ${tokens.spacingHorizontalL}`,
    gap: tokens.spacingHorizontalM,
  },
  leftSection: {
    display: 'flex',
    alignItems: 'center',
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
}

export default function MainLayout({ 
  children, 
  onToggleTheme, 
  isDarkMode,
}: MainLayoutProps) {
  const styles = useStyles()

  return (
    <div className={styles.root}>
      <div className={styles.topBar}>
        <div className={styles.leftSection}>
          <Tooltip content="PyRIT" relationship="label">
            <img 
              src="/roakey.png" 
              alt="Co-PyRIT Logo" 
              className={styles.logo}
            />
          </Tooltip>
          <Text className={styles.title}>Co-PyRIT</Text>
          <Text className={styles.subtitle}>Minimal Demo</Text>
        </div>
        <Button
          appearance="subtle"
          icon={isDarkMode ? <WeatherSunnyRegular /> : <WeatherMoonRegular />}
          onClick={onToggleTheme}
        />
      </div>
      <main className={styles.main}>{children}</main>
    </div>
  )
}
