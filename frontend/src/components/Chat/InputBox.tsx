import { useState, useEffect, useRef, forwardRef, useImperativeHandle, KeyboardEvent } from 'react'
import {
  makeStyles,
  Button,
  tokens,
  Caption1,
  Tooltip,
  Text,
} from '@fluentui/react-components'
import { SendRegular, AttachRegular, DismissRegular, InfoRegular, AddRegular, CopyRegular, WarningRegular, SettingsRegular } from '@fluentui/react-icons'
import { MessageAttachment, TargetInstance } from '../../types'

const useStyles = makeStyles({
  root: {
    padding: `${tokens.spacingVerticalXL} ${tokens.spacingHorizontalXXL}`,
    backgroundColor: tokens.colorNeutralBackground2,
  },
  inputContainer: {
    display: 'flex',
    flexDirection: 'column',
    gap: tokens.spacingVerticalM,
    maxWidth: '900px',
    margin: '0 auto',
  },
  attachmentsContainer: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: tokens.spacingHorizontalS,
    paddingLeft: tokens.spacingHorizontalL,
    paddingRight: tokens.spacingHorizontalL,
    paddingTop: tokens.spacingVerticalS,
  },
  attachmentChip: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXXS,
    padding: `${tokens.spacingVerticalXXS} ${tokens.spacingHorizontalS}`,
    backgroundColor: tokens.colorNeutralBackground4,
    borderRadius: tokens.borderRadiusLarge,
  },
  inputWrapper: {
    position: 'relative',
    display: 'flex',
    flexDirection: 'column',
    backgroundColor: tokens.colorNeutralBackground3,
    borderRadius: '28px',
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
    ':focus-within': {
      borderTopColor: tokens.colorBrandStroke1,
      borderRightColor: tokens.colorBrandStroke1,
      borderBottomColor: tokens.colorBrandStroke1,
      borderLeftColor: tokens.colorBrandStroke1,
      boxShadow: `0 0 0 2px ${tokens.colorBrandBackground2}`,
    },
  },
  inputRow: {
    display: 'flex',
    alignItems: 'center',
    padding: `${tokens.spacingVerticalS} ${tokens.spacingHorizontalL}`,
  },
  textInput: {
    flex: 1,
    backgroundColor: 'transparent',
    border: 'none',
    outline: 'none',
    fontSize: tokens.fontSizeBase300,
    fontFamily: tokens.fontFamilyBase,
    color: tokens.colorNeutralForeground1,
    resize: 'none',
    minHeight: '24px',
    maxHeight: '96px',
    overflowY: 'auto',
    '::placeholder': {
      color: tokens.colorNeutralForeground4,
    },
    '::-webkit-scrollbar': {
      width: '8px',
    },
    '::-webkit-scrollbar-track': {
      backgroundColor: 'transparent',
    },
    '::-webkit-scrollbar-thumb': {
      backgroundColor: tokens.colorNeutralStroke1,
      borderRadius: '4px',
    },
  },
  iconButtonsLeft: {
    display: 'flex',
    gap: tokens.spacingHorizontalXS,
    marginRight: tokens.spacingHorizontalS,
  },
  iconButtonsRight: {
    display: 'flex',
    gap: tokens.spacingHorizontalXS,
    marginLeft: tokens.spacingHorizontalS,
  },
  iconButton: {
    minWidth: '32px',
    width: '32px',
    height: '32px',
    padding: 0,
    borderRadius: '50%',
  },
  sendButton: {
    minWidth: '32px',
    width: '32px',
    height: '32px',
    padding: 0,
    borderRadius: '50%',
  },
  singleTurnWarning: {
    display: 'flex',
    alignItems: 'center',
    color: tokens.colorPaletteYellowForeground2,
  },
  singleTurnBanner: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: tokens.spacingHorizontalM,
    padding: `${tokens.spacingVerticalM} ${tokens.spacingHorizontalL}`,
    backgroundColor: tokens.colorNeutralBackground3,
    borderRadius: '28px',
    border: `1px solid ${tokens.colorNeutralStroke1}`,
  },
  singleTurnText: {
    color: tokens.colorNeutralForeground2,
  },
  noTargetBanner: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: tokens.spacingHorizontalM,
    padding: `${tokens.spacingVerticalM} ${tokens.spacingHorizontalL}`,
    backgroundColor: tokens.colorPaletteRedBackground1,
    borderRadius: '28px',
    border: `1px solid ${tokens.colorPaletteRedBorder1}`,
  },
  noTargetText: {
    color: tokens.colorPaletteRedForeground1,
    fontWeight: tokens.fontWeightSemibold as unknown as string,
  },
})

export interface InputBoxHandle {
  addAttachment: (att: MessageAttachment) => void
  setText: (text: string) => void
}

interface InputBoxProps {
  onSend: (originalValue: string, convertedValue: string | undefined, attachments: MessageAttachment[]) => void
  disabled?: boolean
  activeTarget?: TargetInstance | null
  singleTurnLimitReached?: boolean
  onNewConversation?: () => void
  operatorLocked?: boolean
  crossTargetLocked?: boolean
  onUseAsTemplate?: () => void
  attackOperator?: string
  noTargetSelected?: boolean
  onConfigureTarget?: () => void
}

const InputBox = forwardRef<InputBoxHandle, InputBoxProps>(function InputBox({ onSend, disabled = false, activeTarget, singleTurnLimitReached = false, onNewConversation, operatorLocked = false, crossTargetLocked = false, onUseAsTemplate, attackOperator, noTargetSelected = false, onConfigureTarget }, ref) {
  const styles = useStyles()
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<MessageAttachment[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useImperativeHandle(ref, () => ({
    addAttachment: (att: MessageAttachment) => {
      setAttachments(prev => [...prev, att])
    },
    setText: (text: string) => {
      setInput(text)
    },
  }))

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files) return

    const newAttachments: MessageAttachment[] = []

    for (let i = 0; i < files.length; i++) {
      const file = files[i]
      const url = URL.createObjectURL(file)

      let type: MessageAttachment['type'] = 'file'
      if (file.type.startsWith('image/')) type = 'image'
      else if (file.type.startsWith('audio/')) type = 'audio'
      else if (file.type.startsWith('video/')) type = 'video'

      newAttachments.push({
        type,
        name: file.name,
        url,
        mimeType: file.type,
        size: file.size,
        file,
      })
    }

    setAttachments([...attachments, ...newAttachments])
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const removeAttachment = (index: number) => {
    const newAttachments = [...attachments]
    URL.revokeObjectURL(newAttachments[index].url)
    newAttachments.splice(index, 1)
    setAttachments(newAttachments)
  }

  const handleSend = () => {
    if ((input || attachments.length > 0) && !disabled) {
      onSend(input, undefined, attachments)
      setInput('')
      setAttachments([])
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto'
      }
    }
  }

  // Re-focus the textarea after sending completes (disabled goes false)
  useEffect(() => {
    if (!disabled && textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [disabled])

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Auto-resize textarea whenever input changes (covers paste, setText, etc.)
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 96) + 'px'
    }
  }, [input])

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div className={styles.root}>
      <div className={styles.inputContainer}>
        {noTargetSelected ? (
          <div className={styles.noTargetBanner} data-testid="no-target-banner">
            <WarningRegular fontSize={18} style={{ color: tokens.colorPaletteRedForeground1 }} />
            <Text className={styles.noTargetText} size={300}>
              No target selected
            </Text>
            {onConfigureTarget && (
              <Button
                appearance="primary"
                icon={<SettingsRegular />}
                onClick={onConfigureTarget}
                data-testid="configure-target-input-btn"
              >
                Configure Target
              </Button>
            )}
          </div>
        ) : operatorLocked ? (
          <div className={styles.singleTurnBanner} data-testid="operator-locked-banner">
            <InfoRegular fontSize={18} />
            <Text className={styles.singleTurnText} size={300}>
              This conversation belongs to operator: {attackOperator}.
            </Text>
            {onUseAsTemplate && (
              <Button
                appearance="primary"
                icon={<CopyRegular />}
                onClick={onUseAsTemplate}
                data-testid="use-as-template-btn"
              >
                Continue with your target
              </Button>
            )}
          </div>
        ) : crossTargetLocked ? (
          <div className={styles.singleTurnBanner} data-testid="cross-target-banner">
            <InfoRegular fontSize={18} />
            <Text className={styles.singleTurnText} size={300}>
              This attack uses a different target. Continue with your target to keep the conversation.
            </Text>
            {onUseAsTemplate && (
              <Button
                appearance="primary"
                icon={<CopyRegular />}
                onClick={onUseAsTemplate}
                data-testid="use-as-template-btn"
              >
                Continue with your target
              </Button>
            )}
          </div>
        ) : singleTurnLimitReached ? (
          <div className={styles.singleTurnBanner} data-testid="single-turn-banner">
            <InfoRegular fontSize={18} />
            <Text className={styles.singleTurnText} size={300}>
              This target only supports single-turn conversations.
            </Text>
            {onNewConversation && (
              <Button
                appearance="primary"
                icon={<AddRegular />}
                onClick={onNewConversation}
                data-testid="new-conversation-btn"
              >
                New Conversation
              </Button>
            )}
          </div>
        ) : (
        <>
        <div className={styles.inputWrapper}>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,audio/*,video/*,.pdf,.doc,.docx,.txt"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
          {attachments.length > 0 && (
            <div className={styles.attachmentsContainer}>
              {attachments.map((att, index) => (
                <div key={index} className={styles.attachmentChip}>
                  <Caption1>
                    {att.type === 'image' && '🖼️'}
                    {att.type === 'audio' && '🎵'}
                    {att.type === 'video' && '🎥'}
                    {att.type === 'file' && '📄'}
                    {' '}{att.name} ({formatFileSize(att.size)})
                  </Caption1>
                  <Button
                    appearance="transparent"
                    size="small"
                    icon={<DismissRegular />}
                    onClick={() => removeAttachment(index)}
                  />
                </div>
              ))}
            </div>
          )}
          <div className={styles.inputRow}>
            <div className={styles.iconButtonsLeft}>
            <Button
              className={styles.iconButton}
              appearance="subtle"
              icon={<AttachRegular />}
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled}
              title="Attach files"
            />
            </div>
          <textarea
            ref={textareaRef}
            className={styles.textInput}
            placeholder="Type something here"
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
          />
          <div className={styles.iconButtonsRight}>
            {activeTarget && activeTarget.supports_multiturn_chat === false && (
              <Tooltip
                content="This target does not track conversation history — each turn is sent independently."
                relationship="description"
              >
                <span className={styles.singleTurnWarning}>
                  <InfoRegular fontSize={18} />
                </span>
              </Tooltip>
            )}
            <Button
              className={styles.sendButton}
              appearance="primary"
              icon={<SendRegular />}
              onClick={handleSend}
              disabled={disabled || (!input && attachments.length === 0)}
              title="Send message"
            />
          </div>
          </div>
        </div>
        </>
        )}
      </div>
    </div>
  )
})

export default InputBox
