import { useEffect, useState, useCallback } from 'react'
import {
  makeStyles,
  tokens,
  Button,
  Text,
  Tooltip,
  Badge,
  Spinner,
} from '@fluentui/react-components'
import {
  AddRegular,
  ChatRegular,
  ChatMultipleRegular,
  DismissRegular,
  StarRegular,
  StarFilled,
} from '@fluentui/react-icons'
import { attacksApi } from '../../services/api'
import type { ConversationSummary } from '../../types'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    width: '280px',
    minWidth: '280px',
    borderLeft: `1px solid ${tokens.colorNeutralStroke1}`,
    backgroundColor: tokens.colorNeutralBackground3,
    overflow: 'hidden',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: `${tokens.spacingVerticalS} ${tokens.spacingHorizontalM}`,
    borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
    minHeight: '48px',
    gap: tokens.spacingHorizontalS,
  },
  headerTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
    fontWeight: tokens.fontWeightSemibold,
  },
  conversationList: {
    flex: 1,
    overflowY: 'auto',
    padding: tokens.spacingVerticalXS,
  },
  conversationItem: {
    display: 'flex',
    flexDirection: 'column',
    padding: `${tokens.spacingVerticalS} ${tokens.spacingHorizontalM}`,
    cursor: 'pointer',
    borderRadius: tokens.borderRadiusMedium,
    gap: tokens.spacingVerticalXXS,
    '&:hover': {
      backgroundColor: tokens.colorNeutralBackground1Hover,
    },
  },
  conversationItemActive: {
    backgroundColor: tokens.colorNeutralBackground1Selected,
    '&:hover': {
      backgroundColor: tokens.colorNeutralBackground1Selected,
    },
  },
  conversationHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: tokens.spacingHorizontalXS,
  },
  conversationTitle: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
    overflow: 'hidden',
  },
  preview: {
    color: tokens.colorNeutralForeground3,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  emptyState: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    padding: tokens.spacingVerticalL,
    gap: tokens.spacingVerticalS,
    color: tokens.colorNeutralForeground3,
  },
  loading: {
    display: 'flex',
    justifyContent: 'center',
    padding: tokens.spacingVerticalL,
  },
})

interface ConversationPanelProps {
  attackResultId: string | null
  activeConversationId: string | null
  onSelectConversation: (conversationId: string) => void
  onNewConversation: () => void
  onChangeMainConversation: (conversationId: string) => void
  onClose: () => void
  /** When true, disable mutating actions (new conversation, promote to main) */
  locked?: boolean
  /** Increment to trigger a conversation list refresh (e.g. after sending a message) */
  refreshKey?: number
}

export default function ConversationPanel({
  attackResultId,
  activeConversationId,
  onSelectConversation,
  onNewConversation,
  onChangeMainConversation,
  onClose,
  locked,
  refreshKey,
}: ConversationPanelProps) {
  const styles = useStyles()
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [mainConversationId, setMainConversationId] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  const fetchConversations = useCallback(async () => {
    if (!attackResultId) {
      setConversations([])
      setMainConversationId(null)
      return
    }

    setIsLoading(true)
    try {
      const response = await attacksApi.getConversations(attackResultId)
      setConversations(response.conversations)
      setMainConversationId(response.main_conversation_id)
    } catch {
      setConversations([])
      setMainConversationId(null)
    } finally {
      setIsLoading(false)
    }
  }, [attackResultId])

  useEffect(() => {
    fetchConversations()
  }, [fetchConversations, activeConversationId, refreshKey])

  // Expose refresh via a data attribute on the root element so parent can call it
  // Actually, we'll handle refresh via the attackConversationId dependency

  return (
    <div className={styles.root} data-testid="conversation-panel">
      <div className={styles.header}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: tokens.spacingVerticalXXS, overflow: 'hidden' }}>
          <div className={styles.headerTitle}>
            <ChatMultipleRegular />
            <Text weight="semibold" size={300}>Attack Conversations</Text>
          </div>
          {attackResultId && (
            <Text
              size={100}
              style={{
                color: tokens.colorNeutralForeground3,
                fontFamily: 'monospace',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                paddingLeft: '20px',
              }}
              title={attackResultId}
            >
              {attackResultId}
            </Text>
          )}
        </div>
        <div style={{ display: 'flex', gap: tokens.spacingHorizontalXXS }}>
          <Tooltip content={locked ? 'Cannot modify — attack is locked' : 'New Conversation'} relationship="label">
            <Button
              appearance="subtle"
              size="small"
              icon={<AddRegular />}
              onClick={onNewConversation}
              disabled={!attackResultId || locked}
              data-testid="new-conversation-btn"
            />
          </Tooltip>
          <Tooltip content="Close panel" relationship="label">
            <Button
              appearance="subtle"
              size="small"
              icon={<DismissRegular />}
              onClick={onClose}
              data-testid="close-panel-btn"
            />
          </Tooltip>
        </div>
      </div>

      <div className={styles.conversationList}>
        {isLoading && (
          <div className={styles.loading}>
            <Spinner size="tiny" />
          </div>
        )}

        {!isLoading && conversations.length === 0 && (
          <div className={styles.emptyState}>
            <ChatRegular fontSize={24} />
            <Text size={200}>
              {attackResultId
                ? 'No related conversations'
                : 'Start an attack to see conversations'}
            </Text>
          </div>
        )}

        {!isLoading && conversations.map((conv) => {
          const isActive = conv.conversation_id === activeConversationId
          return (
            <div
              key={conv.conversation_id}
              className={`${styles.conversationItem} ${isActive ? styles.conversationItemActive : ''}`}
              onClick={() => onSelectConversation(conv.conversation_id)}
              data-testid={`conversation-item-${conv.conversation_id}`}
            >
              <div className={styles.conversationHeader}>
                <div className={styles.conversationTitle}>
                  <ChatRegular fontSize={16} />
                  <Text size={200} weight={isActive ? 'semibold' : 'regular'} truncate>
                    {conv.conversation_id}
                  </Text>
                </div>
                <div style={{ display: 'flex', gap: tokens.spacingHorizontalXXS, alignItems: 'center' }}>
                  <Tooltip
                    content={conv.conversation_id === mainConversationId
                      ? 'This is the main conversation.'
                      : 'Promote to main conversation.'}
                    relationship="description"
                  >
                    <Button
                      appearance="subtle"
                      size="small"
                      icon={conv.conversation_id === mainConversationId ? <StarFilled /> : <StarRegular />}
                      disabled={conv.conversation_id === mainConversationId || locked}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (conv.conversation_id !== mainConversationId) {
                          onChangeMainConversation(conv.conversation_id)
                        }
                      }}
                      data-testid={`star-btn-${conv.conversation_id}`}
                      style={{ minWidth: 'auto', padding: '2px' }}
                    />
                  </Tooltip>
                  <Badge appearance="tint" size="small">
                    {conv.message_count}
                  </Badge>
                </div>
              </div>
              {conv.last_message_preview && (
                <Text size={200} className={styles.preview}>
                  {conv.last_message_preview}
                </Text>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export { type ConversationPanelProps }
