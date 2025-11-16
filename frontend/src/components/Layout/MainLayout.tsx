import { ReactNode } from 'react'
import {
  makeStyles,
  tokens,
} from '@fluentui/react-components'
import Navigation from '../Sidebar/Navigation'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    height: '100vh',
    width: '100vw',
    overflow: 'hidden',
  },
  sidebar: {
    width: '280px',
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
}

export default function MainLayout({ children }: MainLayoutProps) {
  const styles = useStyles()

  return (
    <div className={styles.root}>
      <aside className={styles.sidebar}>
        <Navigation />
      </aside>
      <main className={styles.main}>{children}</main>
    </div>
  )
}
