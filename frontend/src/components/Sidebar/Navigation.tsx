import {
  makeStyles,
  Button,
  tokens,
} from '@fluentui/react-components'
import {
  ChatRegular,
  SettingsRegular,
  WeatherMoonRegular,
  WeatherSunnyRegular,
} from '@fluentui/react-icons'

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
  activeButton: {
    width: '44px',
    height: '44px',
    minWidth: '44px',
    padding: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: tokens.colorBrandBackground2,
    borderRadius: tokens.borderRadiusMedium,
  },
  spacer: {
    flex: 1,
  },
})

export type ViewName = 'chat' | 'config'

interface NavigationProps {
  currentView: ViewName
  onNavigate: (view: ViewName) => void
  onToggleTheme: () => void
  isDarkMode: boolean
}

export default function Navigation({ currentView, onNavigate, onToggleTheme, isDarkMode }: NavigationProps) {
  const styles = useStyles()

  return (
    <div className={styles.root}>
      <Button
        className={currentView === 'chat' ? styles.activeButton : styles.iconButton}
        appearance="subtle"
        icon={<ChatRegular />}
        title="Chat"
        onClick={() => onNavigate('chat')}
      />

      <Button
        className={currentView === 'config' ? styles.activeButton : styles.iconButton}
        appearance="subtle"
        icon={<SettingsRegular />}
        title="Configuration"
        onClick={() => onNavigate('config')}
      />

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
