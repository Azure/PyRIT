import { useState, useEffect, useCallback } from 'react'
import {
  makeStyles,
  tokens,
  Text,
  Button,
  Spinner,
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Badge,
} from '@fluentui/react-components'
import { AddRegular, ArrowSyncRegular, CheckmarkRegular } from '@fluentui/react-icons'
import { targetsApi } from '../../services/api'
import type { TargetInstance } from '../../types'
import CreateTargetDialog from './CreateTargetDialog'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    padding: tokens.spacingVerticalXXL,
    overflow: 'auto',
    backgroundColor: tokens.colorNeutralBackground2,
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: tokens.spacingVerticalXL,
  },
  headerLeft: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  headerActions: {
    display: 'flex',
    gap: tokens.spacingHorizontalS,
  },
  tableContainer: {
    flex: 1,
    overflow: 'auto',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: tokens.spacingVerticalXXXL,
    gap: tokens.spacingVerticalM,
  },
  loadingState: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: tokens.spacingVerticalXXXL,
  },
  errorState: {
    padding: tokens.spacingVerticalL,
    color: tokens.colorPaletteRedForeground1,
    textAlign: 'center',
  },
  activeRow: {
    backgroundColor: tokens.colorBrandBackground2,
  },
  targetName: {
    fontFamily: tokens.fontFamilyMonospace,
    fontSize: tokens.fontSizeBase200,
  },
})

interface TargetConfigProps {
  activeTarget: TargetInstance | null
  onSetActiveTarget: (target: TargetInstance) => void
}

export default function TargetConfig({ activeTarget, onSetActiveTarget }: TargetConfigProps) {
  const styles = useStyles()
  const [targets, setTargets] = useState<TargetInstance[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [dialogOpen, setDialogOpen] = useState(false)

  // Retry fetching targets a few times with backoff. The Vite dev proxy
  // returns 502 while the backend is still starting, so a single failed
  // request on initial page load would show a confusing error to the user.
  const fetchTargets = useCallback(async () => {
    const maxRetries = 3
    setLoading(true)
    setError(null)
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const response = await targetsApi.listTargets(200)
        setTargets(response.items)
        setLoading(false)
        return
      } catch (err) {
        if (attempt < maxRetries) {
          // Wait before retrying (1s, 2s, 3s)
          await new Promise(r => setTimeout(r, (attempt + 1) * 1000))
        } else {
          setError(err instanceof Error ? err.message : 'Failed to load targets')
        }
      }
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchTargets()
  }, [fetchTargets])

  const handleTargetCreated = async () => {
    setDialogOpen(false)
    await fetchTargets()
  }

  const isActive = (target: TargetInstance): boolean =>
    activeTarget?.target_registry_name === target.target_registry_name

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <div className={styles.headerLeft}>
          <Text size={600} weight="semibold">Target Configuration</Text>
          <Text size={300} style={{ color: tokens.colorNeutralForeground3 }}>
            Manage targets for attack sessions. Select a target to use in the chat view.
          </Text>
        </div>
        <div className={styles.headerActions}>
          <Button
            appearance="subtle"
            icon={<ArrowSyncRegular />}
            onClick={fetchTargets}
            disabled={loading}
          >
            Refresh
          </Button>
          <Button
            appearance="primary"
            icon={<AddRegular />}
            onClick={() => setDialogOpen(true)}
          >
            New Target
          </Button>
        </div>
      </div>

      {loading && (
        <div className={styles.loadingState}>
          <Spinner label="Loading targets..." />
        </div>
      )}

      {error && (
        <div className={styles.errorState}>
          <Text>Error: {error}</Text>
        </div>
      )}

      {!loading && !error && targets.length === 0 && (
        <div className={styles.emptyState}>
          <Text size={500} weight="semibold">No Targets Configured</Text>
          <Text size={300} style={{ color: tokens.colorNeutralForeground3 }}>
            Add a target manually, or configure an initializer in your <code>~/.pyrit/.pyrit_conf</code> file
            to auto-populate targets from your <code>.env</code> and <code>.env.local</code> files.
            For example, add <code>airt</code> to the <code>initializers</code> list to register
            Azure OpenAI targets automatically. See the{' '}
            <a href="https://github.com/Azure/PyRIT/blob/main/.pyrit_conf_example" target="_blank" rel="noopener noreferrer">
              .pyrit_conf_example
            </a>{' '}
            for details.
          </Text>
          <Button
            appearance="primary"
            icon={<AddRegular />}
            onClick={() => setDialogOpen(true)}
          >
            Create First Target
          </Button>
        </div>
      )}

      {!loading && !error && targets.length > 0 && (
        <div className={styles.tableContainer}>
          <Table aria-label="Target instances">
            <TableHeader>
              <TableRow>
                <TableHeaderCell />
                <TableHeaderCell>Name</TableHeaderCell>
                <TableHeaderCell>Type</TableHeaderCell>
                <TableHeaderCell>Endpoint</TableHeaderCell>
                <TableHeaderCell>Model</TableHeaderCell>
              </TableRow>
            </TableHeader>
            <TableBody>
              {targets.map((target) => (
                <TableRow
                  key={target.target_registry_name}
                  className={isActive(target) ? styles.activeRow : undefined}
                >
                  <TableCell>
                    {isActive(target) ? (
                      <Badge appearance="filled" color="brand" icon={<CheckmarkRegular />}>
                        Active
                      </Badge>
                    ) : (
                      <Button
                        appearance="primary"
                        size="small"
                        onClick={() => onSetActiveTarget(target)}
                      >
                        Set Active
                      </Button>
                    )}
                  </TableCell>
                  <TableCell>
                    <Text className={styles.targetName}>{target.target_registry_name}</Text>
                  </TableCell>
                  <TableCell>
                    <Badge appearance="outline">{target.target_type}</Badge>
                  </TableCell>
                  <TableCell>
                    <Text size={200}>{target.endpoint || '—'}</Text>
                  </TableCell>
                  <TableCell>
                    <Text size={200}>{target.model_name || '—'}</Text>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}

      <CreateTargetDialog
        open={dialogOpen}
        onClose={() => setDialogOpen(false)}
        onCreated={handleTargetCreated}
      />
    </div>
  )
}
