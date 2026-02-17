import { useState } from 'react'
import {
  makeStyles,
  tokens,
  Button,
  Text,
  Badge,
  Tooltip,
} from '@fluentui/react-components'
import { AddRegular } from '@fluentui/react-icons'
import MessageList from './MessageList'
import InputBox from './InputBox'
import { attacksApi } from '../../services/api'
import { buildMessagePieces, backendMessagesToFrontend } from '../../utils/messageMapper'
import type { Message, MessageAttachment, TargetInstance } from '../../types'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    width: '100%',
    backgroundColor: tokens.colorNeutralBackground2,
    overflow: 'hidden',
  },
  ribbon: {
    height: '48px',
    minHeight: '48px',
    flexShrink: 0,
    backgroundColor: tokens.colorNeutralBackground3,
    borderBottom: `1px solid ${tokens.colorNeutralStroke1}`,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: `0 ${tokens.spacingHorizontalL}`,
    gap: tokens.spacingHorizontalM,
  },
  conversationInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalS,
    color: tokens.colorNeutralForeground2,
    fontSize: tokens.fontSizeBase300,
  },
  targetInfo: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXS,
  },
  noTarget: {
    color: tokens.colorNeutralForeground3,
    fontStyle: 'italic',
  },
})

interface ChatWindowProps {
  messages: Message[]
  onSendMessage: (message: Message) => void
  onReceiveMessage: (message: Message) => void
  onNewChat: () => void
  activeTarget: TargetInstance | null
  conversationId: string | null
  onConversationCreated: (id: string) => void
}

export default function ChatWindow({
  messages,
  onSendMessage,
  onReceiveMessage,
  onNewChat,
  activeTarget,
  conversationId,
  onConversationCreated,
}: ChatWindowProps) {
  const styles = useStyles()
  const [isSending, setIsSending] = useState(false)

  const handleSend = async (originalValue: string, _convertedValue: string | undefined, attachments: MessageAttachment[]) => {
    if (!activeTarget) return

    // Add user message with attachments for display
    const userMessage: Message = {
      role: 'user',
      content: originalValue,
      timestamp: new Date().toISOString(),
      attachments: attachments.length > 0 ? attachments : undefined,
    }
    onSendMessage(userMessage)

    // Show loading indicator
    setIsSending(true)
    const loadingMessage: Message = {
      role: 'assistant',
      content: '...',
      timestamp: new Date().toISOString(),
      isLoading: true,
    }
    onReceiveMessage(loadingMessage)

    try {
      // Build message pieces from text + attachments
      const pieces = await buildMessagePieces(originalValue, attachments)

      // Create attack lazily on first message
      let currentConversationId = conversationId
      if (!currentConversationId) {
        const createResponse = await attacksApi.createAttack({
          target_registry_name: activeTarget.target_registry_name,
        })
        currentConversationId = createResponse.conversation_id
        onConversationCreated(currentConversationId)
      }

      // Send message to target
      const response = await attacksApi.addMessage(currentConversationId, {
        role: 'user',
        pieces,
        send: true,
        target_registry_name: activeTarget.target_registry_name,
      })
      // Map backend messages to frontend format
      const backendMessages = backendMessagesToFrontend(response.messages.messages)
      // Replace loading message with the latest assistant response
      const lastAssistantMsg = backendMessages.filter(m => m.role !== 'user').pop()

      if (lastAssistantMsg) {
        // Replace loading with actual response
        onReceiveMessage(lastAssistantMsg)
      }
    } catch (err) {
      // Replace loading with error message
      const errorMessage: Message = {
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        error: {
          type: 'unknown',
          description: err instanceof Error ? err.message : 'Failed to send message',
        },
      }
      onReceiveMessage(errorMessage)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className={styles.root}>
      <div className={styles.ribbon}>
        <div className={styles.conversationInfo}>
          <Text>PyRIT Attack</Text>
          {activeTarget ? (
            <div className={styles.targetInfo}>
              <Text size={200}>→</Text>
              <Tooltip content={activeTarget.target_registry_name} relationship="label">
                <Badge appearance="outline" size="medium">
                  {activeTarget.target_type}
                  {activeTarget.model_name ? ` (${activeTarget.model_name})` : ''}
                </Badge>
              </Tooltip>
            </div>
          ) : (
            <Text size={200} className={styles.noTarget}>
              No target selected — configure one in Settings
            </Text>
          )}
        </div>
        <Button
          appearance="primary"
          icon={<AddRegular />}
          onClick={onNewChat}
        >
          New Chat
        </Button>
      </div>
      <MessageList messages={messages} />
      <InputBox
        onSend={handleSend}
        disabled={isSending || !activeTarget}
      />
    </div>
  )
}
