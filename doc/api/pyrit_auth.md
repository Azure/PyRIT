# pyrit.auth

Authentication functionality for a variety of services.

## Functions

### `get_azure_async_token_provider(scope: str)`

Get an asynchronous Azure token provider using AsyncDefaultAzureCredential.

Returns an async callable suitable for use with async clients like OpenAI's AsyncOpenAI.
The callable handles automatic token refresh.

| Parameter | Type | Description |
|---|---|---|
| `scope` | `str` | The Azure token scope (e.g., 'https://cognitiveservices.azure.com/.default'). |

**Returns:**

- `` — Async callable that returns bearer tokens.

### `get_azure_openai_auth(endpoint: str)`

Get an async Azure token provider for OpenAI endpoints.

Automatically determines the correct scope based on the endpoint URL and returns
an async token provider suitable for use with AsyncOpenAI clients.

| Parameter | Type | Description |
|---|---|---|
| `endpoint` | `str` | The Azure OpenAI endpoint URL. |

**Returns:**

- `` — Async callable that returns bearer tokens.

### `get_azure_token_provider(scope: str) → Callable[[], str]`

Get a synchronous Azure token provider using DefaultAzureCredential.

Returns a callable that returns a bearer token string. The callable handles
automatic token refresh.

| Parameter | Type | Description |
|---|---|---|
| `scope` | `str` | The Azure token scope (e.g., 'https://cognitiveservices.azure.com/.default'). |

**Returns:**

- `Callable[[], str]` — Callable[[], str]: A token provider function that returns bearer tokens.

### `get_default_azure_scope(endpoint: str) → str`

Determine the appropriate Azure token scope based on the endpoint URL.

| Parameter | Type | Description |
|---|---|---|
| `endpoint` | `str` | The Azure endpoint URL. |

**Returns:**

- `str` — The appropriate token scope for the endpoint.
- 'https://ml.azure.com/.default' for AI Foundry endpoints (*.ai.azure.com)
- 'https://cognitiveservices.azure.com/.default' for other Azure endpoints

## `class Authenticator(abc.ABC)`

Abstract base class for authenticators.

**Methods:**

#### `get_token() → str`

Get the current authentication token synchronously.

**Returns:**

- `str` — The current authentication token.

#### `get_token_async() → str`

Get the current authentication token asynchronously.

**Returns:**

- `str` — The current authentication token.

#### `refresh_token() → str`

Refresh the authentication token synchronously.

**Returns:**

- `str` — The refreshed authentication token.

#### `refresh_token_async() → str`

Refresh the authentication token asynchronously.

**Returns:**

- `str` — The refreshed authentication token.

## `class AzureAuth(Authenticator)`

Azure CLI Authentication.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `token_scope` | `str` | The token scope for authentication. |
| `tenant_id` | `str` | The tenant ID. Defaults to "". Defaults to `''`. |

**Methods:**

#### `get_token() → str`

Get the current token.

**Returns:**

- `str` — current token

#### `refresh_token() → str`

Refresh the access token if it is expired.

**Returns:**

- `str` — A token

## `class AzureStorageAuth`

A utility class for Azure Storage authentication, providing methods to generate SAS tokens
using user delegation keys.

**Methods:**

#### `get_sas_token(container_url: str) → str`

Generate a SAS token for the specified blob using a user delegation key.

| Parameter | Type | Description |
|---|---|---|
| `container_url` | `str` | The URL of the Azure Blob Storage container. |

**Returns:**

- `str` — The generated SAS token.

**Raises:**

- `ValueError` — If container_url is empty or invalid.

#### get_user_delegation_key

```python
get_user_delegation_key(blob_service_client: BlobServiceClient) → UserDelegationKey
```

Retrieve a user delegation key valid for one day.

| Parameter | Type | Description |
|---|---|---|
| `blob_service_client` | `BlobServiceClient` | An instance of BlobServiceClient to interact |

**Returns:**

- `UserDelegationKey` — A user delegation key valid for one day.

## `class CopilotAuthenticator(Authenticator)`

Playwright-based authenticator for Microsoft Copilot. Used by WebSocketCopilotTarget.

This authenticator automates browser login to obtain and refresh access tokens that are necessary
for accessing Microsoft Copilot via WebSocket connections. It uses Playwright to simulate user
interactions for authentication, and msal-extensions for encrypted token persistence.

An access token acquired by this authenticator is usually valid for about 60 minutes.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `headless` | `bool` | Whether to run the browser in headless mode. Default is False. Defaults to `False`. |
| `maximized` | `bool` | Whether to start the browser maximized. Default is True. Defaults to `True`. |
| `timeout_for_elements_seconds` | `int` | Timeout used when waiting for page elements, in seconds. Defaults to `DEFAULT_ELEMENT_TIMEOUT_SECONDS`. |
| `token_capture_timeout_seconds` | `int` | Maximum time to wait for token capture via network monitoring. Defaults to `DEFAULT_TOKEN_CAPTURE_TIMEOUT`. |
| `network_retries` | `int` | Number of retry attempts for network operations. Default is 3. Defaults to `DEFAULT_NETWORK_RETRIES`. |
| `fallback_to_plaintext` | `bool` | Whether to fallback to plaintext storage if encryption is unavailable. If set to False (default), an exception will be raised if encryption cannot be used. WARNING: Setting to True stores tokens in plaintext. Defaults to `False`. |

**Methods:**

#### `get_claims() → dict[str, Any]`

Get the JWT claims from the current authentication token.

**Returns:**

- `dict[str, Any]` — dict[str, Any]: The JWT claims decoded from the access token.

#### `get_token_async() → str`

Get the current authentication token.

This checks the cache first and only launches the browser if no valid token is found.
If multiple calls are made concurrently, they will be serialized via an asyncio lock
to prevent launching multiple browser instances.

**Returns:**

- `str` — A valid Bearer token for Microsoft Copilot.

#### `refresh_token_async() → str`

Refresh the authentication token asynchronously.

This will clear the existing token cache and fetch a new token with automated browser login.

**Returns:**

- `str` — The refreshed authentication token.

**Raises:**

- `RuntimeError` — If token refresh fails.

## `class ManualCopilotAuthenticator(Authenticator)`

Simple authenticator that uses a manually-provided access token for Microsoft Copilot.

This authenticator is useful for testing or environments where browser automation is not
possible. Users can obtain the access token from browser DevTools and provide it directly.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `access_token` | `Optional[str]` | A valid JWT access token for Microsoft Copilot. This token can be obtained from browser DevTools when connected to Copilot. If None, the token will be read from the ``COPILOT_ACCESS_TOKEN`` environment variable. Defaults to `None`. |

**Methods:**

#### `get_claims() → dict[str, Any]`

Get the JWT claims from the access token.

**Returns:**

- `dict[str, Any]` — dict[str, Any]: The JWT claims decoded from the access token.

#### `get_token() → str`

Get the current authentication token synchronously.

**Returns:**

- `str` — The access token provided during initialization.

#### `get_token_async() → str`

Get the current authentication token.

**Returns:**

- `str` — The access token provided during initialization.

#### `refresh_token() → str`

Not supported by this authenticator.

**Raises:**

- `RuntimeError` — Always raised as manual tokens cannot be refreshed automatically.

#### `refresh_token_async() → str`

Not supported by this authenticator.

**Raises:**

- `RuntimeError` — Always raised as manual tokens cannot be refreshed automatically.

## `class TokenProviderCredential`

Wrapper to convert a token provider callable into an Azure TokenCredential.

This class bridges the gap between token provider functions (like those returned by
get_azure_token_provider) and Azure SDK clients that require a TokenCredential object.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `token_provider` | `Callable[[], Union[str, Callable[..., Any]]]` | A callable that returns either a token string or an awaitable that returns a token string. |

**Methods:**

#### `get_token(scopes: str = (), kwargs: Any = {}) → AccessToken`

Get an access token.

| Parameter | Type | Description |
|---|---|---|
| `scopes` | `str` | Token scopes (ignored as the scope is already configured in the token provider). Defaults to `()`. |
| `kwargs` | `Any` | Additional arguments (ignored). Defaults to `{}`. |

**Returns:**

- `AccessToken` — The access token with expiration time.
