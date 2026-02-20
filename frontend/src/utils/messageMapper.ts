import type {
  BackendMessage,
  BackendMessagePiece,
  Message,
  MessageAttachment,
  MessageError,
  MessagePieceRequest,
} from '../types'

/**
 * Read a File and return its contents as a base64-encoded string (no data URI prefix).
 */
export function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      // Strip the data:...;base64, prefix
      const base64 = result.split(',')[1] || ''
      resolve(base64)
    }
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

/**
 * Map a frontend MIME type to the backend PromptDataType convention.
 */
export function mimeTypeToDataType(mimeType: string): string {
  if (mimeType.startsWith('image/')) return 'image_path'
  if (mimeType.startsWith('audio/')) return 'audio_path'
  if (mimeType.startsWith('video/')) return 'video_path'
  return 'binary_path'
}

/**
 * Map a backend `converted_value_data_type` to a frontend attachment type.
 */
export function dataTypeToAttachmentType(dataType: string): 'image' | 'audio' | 'video' | 'file' {
  if (dataType.includes('image')) return 'image'
  if (dataType.includes('audio')) return 'audio'
  if (dataType.includes('video')) return 'video'
  return 'file'
}

/**
 * Build a data URI from base64 content and a MIME type.
 */
export function buildDataUri(base64Value: string, mimeType: string): string {
  return `data:${mimeType};base64,${base64Value}`
}

/**
 * Determine a default MIME type for a backend data type when none is provided.
 */
function defaultMimeForDataType(dataType: string): string {
  if (dataType.includes('image')) return 'image/png'
  if (dataType.includes('audio')) return 'audio/wav'
  if (dataType.includes('video')) return 'video/mp4'
  return 'application/octet-stream'
}

/**
 * Check if a backend data type represents non-text media content.
 */
function isMediaDataType(dataType: string): boolean {
  return dataType.includes('image') || dataType.includes('audio') || dataType.includes('video')
}

/**
 * Convert a single backend MessagePiece to a frontend MessageAttachment (or null for text).
 */
function pieceToAttachment(piece: BackendMessagePiece): MessageAttachment | null {
  const dataType = piece.converted_value_data_type
  if (!isMediaDataType(dataType)) return null

  const mime = piece.converted_value_mime_type || defaultMimeForDataType(dataType)
  const isBase64 = !piece.converted_value.startsWith('data:') && !piece.converted_value.startsWith('http')
  const url = isBase64 ? buildDataUri(piece.converted_value, mime) : piece.converted_value

  return {
    type: dataTypeToAttachmentType(dataType),
    name: `${dataType}_${piece.piece_id.slice(0, 8)}`,
    url,
    mimeType: mime,
    size: piece.converted_value.length,
  }
}

/**
 * Extract an error from a backend message piece, if any.
 */
function pieceToError(piece: BackendMessagePiece): MessageError | undefined {
  if (piece.response_error && piece.response_error !== 'none') {
    return {
      type: piece.response_error,
      description: piece.response_error_description || undefined,
    }
  }
  return undefined
}

/**
 * Convert a single backend Message DTO to a frontend Message for rendering.
 */
export function backendMessageToFrontend(msg: BackendMessage): Message {
  const textParts: string[] = []
  const attachments: MessageAttachment[] = []
  let error: MessageError | undefined

  for (const piece of msg.pieces) {
    // Check for errors
    const pieceError = pieceToError(piece)
    if (pieceError && !error) {
      error = pieceError
    }

    // Extract text content from text-type pieces
    if (!isMediaDataType(piece.converted_value_data_type)) {
      if (piece.converted_value) {
        textParts.push(piece.converted_value)
      }
    }

    // Extract media attachments
    const att = pieceToAttachment(piece)
    if (att) {
      attachments.push(att)
    }
  }

  const role = msg.role === 'simulated_assistant' ? 'simulated_assistant'
    : msg.role === 'assistant' ? 'assistant'
    : msg.role === 'system' || msg.role === 'developer' ? 'system'
    : 'user'

  return {
    role: role as Message['role'],
    content: textParts.join('\n'),
    timestamp: msg.created_at,
    attachments: attachments.length > 0 ? attachments : undefined,
    error,
  }
}

/**
 * Convert all backend messages to frontend messages.
 */
export function backendMessagesToFrontend(messages: BackendMessage[]): Message[] {
  return messages.map(backendMessageToFrontend)
}

/**
 * Convert a frontend MessageAttachment (with File) to a backend MessagePieceRequest.
 */
export async function attachmentToMessagePieceRequest(att: MessageAttachment): Promise<MessagePieceRequest> {
  let base64Value = ''
  if (att.file) {
    base64Value = await fileToBase64(att.file)
  } else if (att.url.startsWith('data:')) {
    base64Value = att.url.split(',')[1] || ''
  } else {
    base64Value = att.url
  }

  return {
    data_type: mimeTypeToDataType(att.mimeType),
    original_value: base64Value,
    mime_type: att.mimeType,
  }
}

/**
 * Build the pieces array for an AddMessageRequest from text + attachments.
 */
export async function buildMessagePieces(
  text: string,
  attachments: MessageAttachment[]
): Promise<MessagePieceRequest[]> {
  const pieces: MessagePieceRequest[] = []

  // Add text piece if present
  if (text.trim()) {
    pieces.push({
      data_type: 'text',
      original_value: text,
    })
  }

  // Add attachment pieces
  for (const att of attachments) {
    pieces.push(await attachmentToMessagePieceRequest(att))
  }

  return pieces
}
