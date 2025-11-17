import { useState, useEffect } from 'react'
import {
  makeStyles,
  Text,
  tokens,
  Table,
  TableHeader,
  TableRow,
  TableHeaderCell,
  TableBody,
  TableCell,
  Button,
  Spinner,
} from '@fluentui/react-components'
import { OpenRegular } from '@fluentui/react-icons'

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
    marginBottom: tokens.spacingVerticalXL,
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalXS,
  },
  title: {
    fontSize: tokens.fontSizeHero700,
    fontWeight: tokens.fontWeightSemibold,
  },
  tableContainer: {
    backgroundColor: tokens.colorNeutralBackground1,
    borderRadius: tokens.borderRadiusLarge,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    overflow: 'auto',
  },
  clickableRow: {
    cursor: 'pointer',
    ':hover': {
      backgroundColor: tokens.colorNeutralBackground1Hover,
    },
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    padding: tokens.spacingVerticalXXXL,
    gap: tokens.spacingVerticalM,
  },
  loadingContainer: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: tokens.spacingVerticalXXXL,
  },
  actionButton: {
    minWidth: 'auto',
  },
})

interface ConversationRow {
  conversation_id: string
  first_prompt: string
  message_count: number
  labels: Record<string, string>
  metadata: Record<string, any>
  created_at: string
  updated_at: string
}

interface HistoryPageProps {
  onSelectConversation: (conversationId: string) => void
}

export default function HistoryPage({ onSelectConversation }: HistoryPageProps) {
  const styles = useStyles()
  const [conversations, setConversations] = useState<ConversationRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadConversations()
  }, [])

  const loadConversations = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch('/api/chat/conversations')
      if (!response.ok) {
        throw new Error('Failed to load conversations')
      }
      const data = await response.json()
      
      // Transform the data to get first prompt and message count
      const rows: ConversationRow[] = data.map((conv: any) => {
        const firstUserMessage = conv.messages?.find((m: any) => m.role === 'user')
        const messageCount = conv.messages?.length || 0
        
        return {
          conversation_id: conv.conversation_id,
          first_prompt: firstUserMessage?.content || '(No messages)',
          message_count: messageCount,
          labels: {}, // Backend doesn't provide labels yet
          metadata: {}, // Backend doesn't provide metadata yet
          created_at: conv.created_at,
          updated_at: conv.updated_at,
        }
      })
      
      setConversations(rows)
    } catch (err) {
      console.error('Error loading conversations:', err)
      setError(err instanceof Error ? err.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const handleRowClick = (conversationId: string) => {
    onSelectConversation(conversationId)
  }

  const formatDate = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleString()
    } catch {
      return dateString
    }
  }

  const truncateText = (text: string, maxLength: number = 100) => {
    if (text.length <= maxLength) return text
    return text.substring(0, maxLength) + '...'
  }

  if (loading) {
    return (
      <div className={styles.root}>
        <div className={styles.loadingContainer}>
          <Spinner label="Loading conversations..." />
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={styles.root}>
        <div className={styles.emptyState}>
          <Text size={500} weight="semibold">Error loading conversations</Text>
          <Text>{error}</Text>
          <Button appearance="primary" onClick={loadConversations}>Retry</Button>
        </div>
      </div>
    )
  }

  if (conversations.length === 0) {
    return (
      <div className={styles.root}>
        <div className={styles.header}>
          <Text className={styles.title}>Conversation History</Text>
        </div>
        <div className={styles.emptyState}>
          <Text size={500} weight="semibold">No conversations yet</Text>
          <Text style={{ color: tokens.colorNeutralForeground3 }}>
            Start a conversation to see it here
          </Text>
        </div>
      </div>
    )
  }

  return (
    <div className={styles.root}>
      <div className={styles.header}>
        <Text className={styles.title}>Conversation History</Text>
        <Text style={{ color: tokens.colorNeutralForeground3 }}>
          {conversations.length} conversation{conversations.length !== 1 ? 's' : ''}
        </Text>
      </div>

      <div className={styles.tableContainer}>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHeaderCell>Conversation ID</TableHeaderCell>
              <TableHeaderCell>First Prompt</TableHeaderCell>
              <TableHeaderCell>Messages</TableHeaderCell>
              <TableHeaderCell>Created</TableHeaderCell>
              <TableHeaderCell>Updated</TableHeaderCell>
              <TableHeaderCell>Action</TableHeaderCell>
            </TableRow>
          </TableHeader>
          <TableBody>
            {conversations.map((conv) => (
              <TableRow 
                key={conv.conversation_id}
                className={styles.clickableRow}
                onClick={() => handleRowClick(conv.conversation_id)}
              >
                <TableCell>
                  <Text size={200} style={{ fontFamily: 'monospace' }}>
                    {conv.conversation_id.substring(0, 64)}...
                  </Text>
                </TableCell>
                <TableCell>
                  <Text>{truncateText(conv.first_prompt, 80)}</Text>
                </TableCell>
                <TableCell>
                  <Text>{conv.message_count}</Text>
                </TableCell>
                <TableCell>
                  <Text size={200}>{formatDate(conv.created_at)}</Text>
                </TableCell>
                <TableCell>
                  <Text size={200}>{formatDate(conv.updated_at)}</Text>
                </TableCell>
                <TableCell>
                  <Button
                    className={styles.actionButton}
                    appearance="subtle"
                    icon={<OpenRegular />}
                    onClick={(e) => {
                      e.stopPropagation()
                      handleRowClick(conv.conversation_id)
                    }}
                    title="Open conversation"
                  />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
