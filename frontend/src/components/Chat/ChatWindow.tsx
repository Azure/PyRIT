import { useState } from 'react'
import {
  makeStyles,
  tokens,
  Button,
  Text,
} from '@fluentui/react-components'
import { AddRegular } from '@fluentui/react-icons'
import MessageList from './MessageList'
import InputBox from './InputBox'
import ConverterDrawer from './ConverterDrawer'
import { Message, MessageAttachment } from '../../types'
import { chatApi } from '../../services/api'

const useStyles = makeStyles({
  root: {
    display: 'flex',
    flexDirection: 'row',
    height: '100%',
    width: '100%',
    backgroundColor: tokens.colorNeutralBackground2,
    position: 'relative',
    overflow: 'hidden',
  },
  chatContainer: {
    display: 'flex',
    flexDirection: 'column',
    height: '100%',
    minWidth: 0,
    flex: '1 1 auto',
    transition: 'all 0.3s ease-in-out',
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
  converterContainer: {
    height: '100%',
    flex: '0 0 auto',
    transition: 'all 0.3s ease-in-out',
    overflow: 'hidden',
    borderRight: `1px solid ${tokens.colorNeutralStroke1}`,
  },
})

interface ChatWindowProps {
  messages: Message[]
  conversationId: string | null
  onSendMessage: (message: Message) => void
  onReceiveMessage: (message: Message, conversationId: string) => void
  onNewChat: () => void
}

export default function ChatWindow({
  messages,
  conversationId,
  onSendMessage,
  onReceiveMessage,
  onNewChat,
}: ChatWindowProps) {
  const styles = useStyles()
  const [isSending, setIsSending] = useState(false)
  const [isConverterOpen, setIsConverterOpen] = useState(false)
  const [inputText, setInputText] = useState('')
  const [onApplyCallback, setOnApplyCallback] = useState<((text: string, identifiers: Array<Record<string, string>>) => void) | null>(null)

  const handleConverterApply = (convertedText: string, _converters: any[], identifiers: Array<Record<string, string>>) => {
    // Call back to InputBox to update its state
    if (onApplyCallback) {
      onApplyCallback(convertedText, identifiers)
    }
  }

  const handleSend = async (originalValue: string, convertedValue: string | undefined, attachments: MessageAttachment[], converterIdentifiers?: Array<Record<string, string>>) => {
    // Add user message with attachments for display (show converted if available)
    const displayText = convertedValue || originalValue
    const userMessage: Message = {
      role: 'user',
      content: displayText,
      timestamp: new Date().toISOString(),
      attachments: attachments.length > 0 ? attachments : undefined,
    }
    onSendMessage(userMessage)

    // Add temporary loading message
    const loadingMessage: Message = {
      role: 'assistant',
      content: '...',  // Will be styled as animated ellipsis
      timestamp: new Date().toISOString(),
    }
    onSendMessage(loadingMessage)

    // Send to API
    setIsSending(true)
    try {
      const response = await chatApi.sendMessage({
        original_value: originalValue,
        converted_value: convertedValue,
        conversation_id: conversationId || undefined,
        attachments: attachments,
        converter_identifiers: converterIdentifiers,
      })

      // Remove loading message and add real response
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.message,
        timestamp: response.timestamp,
      }
      // This will replace the loading message since we're using the callback
      onReceiveMessage(assistantMessage, response.conversation_id)
    } catch (error) {
      console.error('Failed to send message:', error)
      // Replace loading message with detailed error
      const errorDetails = error instanceof Error ? error.message : String(error)
      const errorMessage: Message = {
        role: 'assistant',
        content: `❌ Error: ${errorDetails}\n\nPlease check the browser console for more details.`,
        timestamp: new Date().toISOString(),
      }
      onReceiveMessage(errorMessage, conversationId || '')
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className={styles.root}>
      {isConverterOpen && (
        <div className={styles.converterContainer} style={{ flexBasis: '40%' }}>
          <ConverterDrawer
            isOpen={isConverterOpen}
            onClose={() => setIsConverterOpen(false)}
            onApply={handleConverterApply}
            initialText={inputText}
          />
        </div>
      )}
      <div className={styles.chatContainer} style={{ flexBasis: isConverterOpen ? '60%' : '100%' }}>
        <div className={styles.ribbon}>
          <div className={styles.conversationInfo}>
            <Text>
              {conversationId ? `Conversation: ${conversationId}` : 'New Conversation'}
            </Text>
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
          disabled={isSending}
          onConverterToggle={(isOpen, text) => {
            setIsConverterOpen(isOpen)
            setInputText(text)
          }}
          registerApplyCallback={(callback: (text: string, identifiers: Array<Record<string, string>>) => void) => {
            setOnApplyCallback(() => callback)
          }}
        />
      </div>
    </div>
  )
}
