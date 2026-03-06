import { useState, useEffect, useCallback } from 'react'
import {
  makeStyles,
  tokens,
  Text,
  Button,
  Badge,
  Tooltip,
  Dropdown,
  Option,
  Combobox,
  Spinner,
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableHeaderCell,
  TableRow,
  MessageBar,
  MessageBarBody,
} from '@fluentui/react-components'
import {
  ArrowSyncRegular,
  ChevronLeftRegular,
  ChevronRightRegular,
  OpenRegular,
  CheckmarkCircleRegular,
  DismissCircleRegular,
  QuestionCircleRegular,
  FilterRegular,
} from '@fluentui/react-icons'
import { attacksApi, labelsApi } from '../../services/api'
import { toApiError } from '../../services/errors'
import type { AttackSummary } from '../../types'


const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    overflow: 'hidden',
    backgroundColor: tokens.colorNeutralBackground2,
  },
  header: {
    padding: `${tokens.spacingVerticalM} ${tokens.spacingHorizontalXXL}`,
    borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
    backgroundColor: tokens.colorNeutralBackground3,
  },
  headerRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: tokens.spacingVerticalS,
  },
  filters: {
    display: 'flex',
    gap: tokens.spacingHorizontalS,
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  filterDropdown: {
    minWidth: '160px',
  },
  content: {
    flex: 1,
    overflowY: 'auto',
    overflowX: 'auto',
  },
  table: {
    minWidth: '100%',
    tableLayout: 'auto' as const,
  },
  colStatus: { minWidth: '100px', whiteSpace: 'nowrap' as const },
  colAttackType: { minWidth: '110px', whiteSpace: 'nowrap' as const },
  colTarget: { minWidth: '120px', whiteSpace: 'nowrap' as const },
  colOperator: { minWidth: '120px', whiteSpace: 'nowrap' as const },
  colOperation: { minWidth: '130px', whiteSpace: 'nowrap' as const },
  colMessages: { minWidth: '60px', whiteSpace: 'nowrap' as const },
  colConversations: { minWidth: '70px', whiteSpace: 'nowrap' as const },
  colConverters: { minWidth: '120px', whiteSpace: 'nowrap' as const },
  colLabels: { minWidth: '180px', whiteSpace: 'nowrap' as const },
  colDate: { minWidth: '100px', whiteSpace: 'nowrap' as const },
  colAction: { minWidth: '48px', whiteSpace: 'nowrap' as const },
  clickableRow: {
    cursor: 'pointer',
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground1Hover,
    },
  },
  previewCell: {
    display: 'block',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    maxWidth: '400px',
  },
  nowrap: {
    whiteSpace: 'nowrap',
  },
  badgeGroup: {
    display: 'flex',
    gap: tokens.spacingHorizontalXXS,
    flexWrap: 'wrap',
  },
  pagination: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: tokens.spacingHorizontalM,
    padding: `${tokens.spacingVerticalS} ${tokens.spacingHorizontalXXL}`,
    borderTop: `1px solid ${tokens.colorNeutralStroke1}`,
    backgroundColor: tokens.colorNeutralBackground3,
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    gap: tokens.spacingVerticalM,
    padding: tokens.spacingVerticalXXXL,
    color: tokens.colorNeutralForeground3,
  },
})

const OUTCOME_ICONS: Record<string, React.ReactElement> = {
  success: <CheckmarkCircleRegular style={{ color: tokens.colorPaletteGreenForeground1 }} />,
  failure: <DismissCircleRegular style={{ color: tokens.colorPaletteRedForeground1 }} />,
  undetermined: <QuestionCircleRegular style={{ color: tokens.colorNeutralForeground3 }} />,
}

const OUTCOME_COLORS: Record<string, 'success' | 'danger' | 'informative'> = {
  success: 'success',
  failure: 'danger',
  undetermined: 'informative',
}

interface AttackHistoryProps {
  onOpenAttack: (attackResultId: string) => void
}

export default function AttackHistory({ onOpenAttack }: AttackHistoryProps) {
  const styles = useStyles()
  const [attacks, setAttacks] = useState<AttackSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Filter state
  const [attackClassFilter, setAttackClassFilter] = useState<string>('')
  const [outcomeFilter, setOutcomeFilter] = useState<string>('')
  const [converterFilter, setConverterFilter] = useState<string>('')
  const [operatorFilter, setOperatorFilter] = useState<string>('')
  const [operationFilter, setOperationFilter] = useState<string>('')
  const [otherLabelFilters, setOtherLabelFilters] = useState<string[]>([])
  const [labelSearchText, setLabelSearchText] = useState<string>('')

  // Filter options
  const [attackClassOptions, setAttackClassOptions] = useState<string[]>([])
  const [converterOptions, setConverterOptions] = useState<string[]>([])
  const [operatorOptions, setOperatorOptions] = useState<string[]>([])
  const [operationOptions, setOperationOptions] = useState<string[]>([])
  const [otherLabelOptions, setOtherLabelOptions] = useState<string[]>([])

  // Pagination
  const [cursor, setCursor] = useState<string | undefined>(undefined)
  const [hasMore, setHasMore] = useState(false)
  const [page, setPage] = useState(0)

  const PAGE_SIZE = 25

  const fetchAttacks = useCallback(async (pageCursor?: string) => {
    setLoading(true)
    setError(null)
    try {
      const params: Record<string, unknown> = { limit: PAGE_SIZE }
      if (pageCursor) params.cursor = pageCursor
      if (attackClassFilter) params.attack_type = attackClassFilter
      if (outcomeFilter) params.outcome = outcomeFilter
      if (converterFilter) params.converter_types = [converterFilter]
      const labelParams: string[] = []
      if (operatorFilter) labelParams.push(`operator:${operatorFilter}`)
      if (operationFilter) labelParams.push(`operation:${operationFilter}`)
      labelParams.push(...otherLabelFilters)
      if (labelParams.length > 0) params.label = labelParams

      const response = await attacksApi.listAttacks(params as Parameters<typeof attacksApi.listAttacks>[0])
      setAttacks(response.items.map(attack => ({ ...attack, labels: attack.labels ?? {} })))
      setHasMore(response.pagination.has_more)
      setCursor(response.pagination.next_cursor ?? undefined)
    } catch (err) {
      setAttacks([])
      setError(toApiError(err).detail)
    } finally {
      setLoading(false)
    }
  }, [attackClassFilter, outcomeFilter, converterFilter, operatorFilter, operationFilter, otherLabelFilters])

  // Load filter options on mount
  useEffect(() => {
    attacksApi.getAttackOptions()
      .then(resp => setAttackClassOptions(resp.attack_types))
      .catch(() => { /* ignore */ })
    attacksApi.getConverterOptions()
      .then(resp => setConverterOptions(resp.converter_types))
      .catch(() => { /* ignore */ })
    labelsApi.getLabels()
      .then(resp => {
        const operators: string[] = []
        const operations: string[] = []
        const others: string[] = []
        for (const [key, values] of Object.entries(resp.labels)) {
          if (key === 'operator') {
            operators.push(...values)
          } else if (key === 'operation') {
            operations.push(...values)
          } else if (key !== 'source') {
            for (const val of values) {
              others.push(`${key}:${val}`)
            }
          }
        }
        setOperatorOptions(operators.sort())
        setOperationOptions(operations.sort())
        setOtherLabelOptions(others.sort())
      })
      .catch(() => { /* ignore */ })
  }, [])

  // Reload when filters change
  useEffect(() => {
    setPage(0)
    setCursor(undefined)
    fetchAttacks()
  }, [fetchAttacks])

  const handleNextPage = () => {
    if (cursor) {
      setPage(p => p + 1)
      fetchAttacks(cursor)
    }
  }

  const handlePrevPage = () => {
    setPage(0)
    setCursor(undefined)
    fetchAttacks()
  }

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <div className={styles.headerRow}>
          <Text size={500} weight="semibold">Attack History</Text>
          <Button
            appearance="subtle"
            icon={<ArrowSyncRegular />}
            onClick={() => fetchAttacks()}
            disabled={loading}
            data-testid="refresh-btn"
          >
            Refresh
          </Button>
        </div>
        <div className={styles.filters}>
          <FilterRegular />
          <Dropdown
            className={styles.filterDropdown}
            placeholder="All attack types"
            value={attackClassFilter || undefined}
            selectedOptions={attackClassFilter ? [attackClassFilter] : []}
            onOptionSelect={(_e, data) => setAttackClassFilter(data.optionValue ?? '')}
            data-testid="attack-class-filter"
          >
            <Option value="">All attack types</Option>
            {attackClassOptions.map(cls => (
              <Option key={cls} value={cls}>{cls}</Option>
            ))}
          </Dropdown>
          <Dropdown
            className={styles.filterDropdown}
            placeholder="All outcomes"
            value={outcomeFilter || undefined}
            selectedOptions={outcomeFilter ? [outcomeFilter] : []}
            onOptionSelect={(_e, data) => setOutcomeFilter(data.optionValue ?? '')}
            data-testid="outcome-filter"
          >
            <Option value="">All outcomes</Option>
            <Option value="success">Success</Option>
            <Option value="failure">Failure</Option>
            <Option value="undetermined">Undetermined</Option>
          </Dropdown>
          <Dropdown
            className={styles.filterDropdown}
            placeholder="All converters"
            value={converterFilter || undefined}
            selectedOptions={converterFilter ? [converterFilter] : []}
            onOptionSelect={(_e, data) => setConverterFilter(data.optionValue ?? '')}
            data-testid="converter-filter"
          >
            <Option value="">All converters</Option>
            {converterOptions.map(c => (
              <Option key={c} value={c}>{c}</Option>
            ))}
          </Dropdown>
          <Dropdown
            className={styles.filterDropdown}
            placeholder="All operators"
            value={operatorFilter || undefined}
            selectedOptions={operatorFilter ? [operatorFilter] : []}
            onOptionSelect={(_e, data) => setOperatorFilter(data.optionValue ?? '')}
            data-testid="operator-filter"
          >
            <Option value="">All operators</Option>
            {operatorOptions.map(o => (
              <Option key={o} value={o}>{o}</Option>
            ))}
          </Dropdown>
          <Dropdown
            className={styles.filterDropdown}
            placeholder="All operations"
            value={operationFilter || undefined}
            selectedOptions={operationFilter ? [operationFilter] : []}
            onOptionSelect={(_e, data) => setOperationFilter(data.optionValue ?? '')}
            data-testid="operation-filter"
          >
            <Option value="">All operations</Option>
            {operationOptions.map(o => (
              <Option key={o} value={o}>{o}</Option>
            ))}
          </Dropdown>
          <Combobox
            className={styles.filterDropdown}
            placeholder="Filter labels..."
            multiselect
            selectedOptions={otherLabelFilters}
            onOptionSelect={(_e, data) => {
              setOtherLabelFilters(data.selectedOptions)
              setLabelSearchText('')
            }}
            value={labelSearchText}
            onChange={(e) => setLabelSearchText((e.target as HTMLInputElement).value)}
            data-testid="label-filter"
            freeform
          >
            {otherLabelOptions
              .filter(l => !labelSearchText || l.toLowerCase().includes(labelSearchText.toLowerCase()))
              .slice(0, 50)
              .map(l => (
                <Option key={l} value={l}>{l}</Option>
              ))}
            {otherLabelOptions.filter(l => !labelSearchText || l.toLowerCase().includes(labelSearchText.toLowerCase())).length > 50 && (
              <Option disabled value="__more" text={`Type to search more...`}>{`Type to search ${otherLabelOptions.length - 50} more...`}</Option>
            )}
          </Combobox>
        </div>
      </div>

      <div className={styles.content}>
        {loading ? (
          <div className={styles.emptyState}>
            <Spinner size="medium" label="Loading attacks..." />
          </div>
        ) : error ? (
          <div className={styles.emptyState} data-testid="error-state">
            <MessageBar intent="error">
              <MessageBarBody>{error}</MessageBarBody>
            </MessageBar>
            <Button
              appearance="primary"
              icon={<ArrowSyncRegular />}
              onClick={() => fetchAttacks()}
              disabled={loading}
              data-testid="retry-btn"
            >
              Retry
            </Button>
          </div>
        ) : attacks.length === 0 ? (
          <div className={styles.emptyState} data-testid="empty-state">
            <Text size={400}>No attacks found</Text>
            <Text size={200}>
              {attackClassFilter || outcomeFilter || converterFilter || operatorFilter || operationFilter || otherLabelFilters.length > 0
                ? 'Try adjusting your filters.'
                : 'Run an attack to see it here.'}
            </Text>
          </div>
        ) : (
          <Table className={styles.table} data-testid="attacks-table">
            <TableHeader>
              <TableRow>
                <TableHeaderCell className={styles.colStatus}>Status</TableHeaderCell>
                <TableHeaderCell className={styles.colAttackType}>Attack Type</TableHeaderCell>
                <TableHeaderCell className={styles.colTarget}>Target</TableHeaderCell>
                <TableHeaderCell className={styles.colOperator}>Operator</TableHeaderCell>
                <TableHeaderCell className={styles.colOperation}>Operation</TableHeaderCell>
                <TableHeaderCell className={styles.colMessages}>Msgs</TableHeaderCell>
                <TableHeaderCell className={styles.colConversations}>Convs</TableHeaderCell>
                <TableHeaderCell className={styles.colConverters}>Converters</TableHeaderCell>
                <TableHeaderCell className={styles.colLabels}>Labels</TableHeaderCell>
                <TableHeaderCell className={styles.colDate}>Created</TableHeaderCell>
                <TableHeaderCell className={styles.colDate}>Updated</TableHeaderCell>
                <TableHeaderCell>Last Message</TableHeaderCell>
                <TableHeaderCell className={styles.colAction} />
              </TableRow>
            </TableHeader>
            <TableBody>
              {attacks.map(attack => (
                <TableRow
                  key={attack.attack_result_id}
                  className={styles.clickableRow}
                  onClick={() => onOpenAttack(attack.attack_result_id)}
                  data-testid={`attack-row-${attack.attack_result_id}`}
                >
                  <TableCell>
                    <Badge
                      appearance="filled"
                      color={OUTCOME_COLORS[attack.outcome ?? 'undetermined'] ?? 'informative'}
                      icon={OUTCOME_ICONS[attack.outcome ?? 'undetermined']}
                      data-testid={`outcome-badge-${attack.attack_result_id}`}
                    >
                      {attack.outcome ?? 'undetermined'}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <Text size={200} weight="semibold" truncate>{attack.attack_type}</Text>
                  </TableCell>
                  <TableCell>
                    {attack.target ? (
                      <Tooltip content={`${attack.target.target_type}${attack.target.model_name ? ` (${attack.target.model_name})` : ''}`} relationship="label">
                        <Badge appearance="outline" size="small">
                          {attack.target.model_name || attack.target.target_type}
                        </Badge>
                      </Tooltip>
                    ) : (
                      <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>—</Text>
                    )}
                  </TableCell>
                  <TableCell>
                    <Text size={200} className={styles.nowrap}>{attack.labels.operator || '—'}</Text>
                  </TableCell>
                  <TableCell>
                    <Text size={200} className={styles.nowrap}>{attack.labels.operation || '—'}</Text>
                  </TableCell>
                  <TableCell>
                    <Text size={200}>{attack.message_count}</Text>
                  </TableCell>
                  <TableCell>
                    <Text size={200}>{(attack.related_conversation_ids?.length ?? 0) + 1}</Text>
                  </TableCell>
                  <TableCell>
                    {attack.converters.length > 0 ? (
                      <div className={styles.badgeGroup}>
                        {attack.converters.slice(0, 2).map(c => (
                          <Badge key={c} appearance="tint" size="small">{c}</Badge>
                        ))}
                        {attack.converters.length > 2 && (
                          <Tooltip content={attack.converters.join(', ')} relationship="label">
                            <Badge appearance="tint" size="small">+{attack.converters.length - 2}</Badge>
                          </Tooltip>
                        )}
                      </div>
                    ) : (
                      <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>—</Text>
                    )}
                  </TableCell>
                  <TableCell>
                    {(() => {
                      const otherLabels = Object.entries(attack.labels ?? {}).filter(([k]) => k !== 'operator' && k !== 'operation' && k !== 'source')
                      return otherLabels.length > 0 ? (
                        <div className={styles.badgeGroup}>
                          {otherLabels.slice(0, 2).map(([k, v]) => (
                            <Badge key={k} appearance="tint" size="small" color="brand">{k}: {v}</Badge>
                          ))}
                          {otherLabels.length > 2 && (
                            <Tooltip
                              content={otherLabels.map(([k, v]) => `${k}: ${v}`).join(', ')}
                              relationship="label"
                            >
                              <Badge appearance="tint" size="small" color="brand">
                                +{otherLabels.length - 2}
                              </Badge>
                            </Tooltip>
                          )}
                        </div>
                      ) : (
                        <Text size={200} style={{ color: tokens.colorNeutralForeground3 }}>—</Text>
                      )
                    })()}
                  </TableCell>
                  <TableCell>
                    <Text size={200} className={styles.nowrap}>{formatDate(attack.created_at)}</Text>
                  </TableCell>
                  <TableCell>
                    <Text size={200} className={styles.nowrap}>{formatDate(attack.updated_at)}</Text>
                  </TableCell>
                  <TableCell>
                    <Text size={200} className={styles.previewCell}>
                      {attack.last_message_preview || '—'}
                    </Text>
                  </TableCell>
                  <TableCell>
                    <Tooltip content="Open attack" relationship="label">
                      <Button
                        appearance="subtle"
                        size="small"
                        icon={<OpenRegular />}
                        onClick={(e) => {
                          e.stopPropagation()
                          onOpenAttack(attack.attack_result_id)
                        }}
                        data-testid={`open-attack-${attack.attack_result_id}`}
                      />
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>

      {!loading && attacks.length > 0 && (
        <div className={styles.pagination}>
          <Button
            appearance="subtle"
            icon={<ChevronLeftRegular />}
            disabled={page === 0}
            onClick={handlePrevPage}
            data-testid="prev-page-btn"
          >
            First
          </Button>
          <Text size={200}>Page {page + 1}</Text>
          <Button
            appearance="subtle"
            icon={<ChevronRightRegular />}
            iconPosition="after"
            disabled={!hasMore}
            onClick={handleNextPage}
            data-testid="next-page-btn"
          >
            Next
          </Button>
        </div>
      )}
    </div>
  )
}
