import {
  makeStyles,
  Button,
  tokens,
} from '@fluentui/react-components'
import {
  ChatRegular,
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

  return (
    <div className={styles.root}>
      <Button
        className={styles.iconButton}
        appearance="subtle"
        icon={<ChatRegular />}
        title="Chat"
        disabled
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
