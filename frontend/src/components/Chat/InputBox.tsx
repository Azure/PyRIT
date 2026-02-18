import { useState, KeyboardEvent, useRef } from 'react'
import {
  makeStyles,
  Button,
  tokens,
  Caption1,
} from '@fluentui/react-components'
import { SendRegular, AttachRegular, DismissRegular } from '@fluentui/react-icons'
import { MessageAttachment } from '../../types'

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
    marginBottom: tokens.spacingVerticalS,
  },
  attachmentChip: {
    display: 'flex',
    alignItems: 'center',
    gap: tokens.spacingHorizontalXXS,
    padding: `${tokens.spacingVerticalXXS} ${tokens.spacingHorizontalS}`,
    backgroundColor: tokens.colorNeutralBackground3,
    borderRadius: tokens.borderRadiusLarge,
    border: `1px solid ${tokens.colorNeutralStroke1}`,
  },
  inputWrapper: {
    position: 'relative',
    display: 'flex',
    alignItems: 'center',
    backgroundColor: tokens.colorNeutralBackground3,
    borderRadius: '28px',
    border: `1px solid ${tokens.colorNeutralStroke1}`,
    padding: `${tokens.spacingVerticalS} ${tokens.spacingHorizontalL}`,
    transition: 'border-color 0.2s ease, box-shadow 0.2s ease',
    ':focus-within': {
      borderTopColor: tokens.colorBrandStroke1,
      borderRightColor: tokens.colorBrandStroke1,
      borderBottomColor: tokens.colorBrandStroke1,
      borderLeftColor: tokens.colorBrandStroke1,
      boxShadow: `0 0 0 2px ${tokens.colorBrandBackground2}`,
    },
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
    maxHeight: '120px',
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
})

interface InputBoxProps {
  onSend: (originalValue: string, convertedValue: string | undefined, attachments: MessageAttachment[]) => void
  disabled?: boolean
}

export default function InputBox({ onSend, disabled = false }: InputBoxProps) {
  const styles = useStyles()
  const [input, setInput] = useState('')
  const [attachments, setAttachments] = useState<MessageAttachment[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

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

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleInput = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value)
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px'
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + ' B'
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  }

  return (
    <div className={styles.root}>
      <div className={styles.inputContainer}>
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
        <div className={styles.inputWrapper}>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*,audio/*,video/*,.pdf,.doc,.docx,.txt"
            style={{ display: 'none' }}
            onChange={handleFileSelect}
          />
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
    </div>
  )
}
