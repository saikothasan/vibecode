import { OpenAI } from 'openai';
import { Stream } from 'openai/streaming';
import { z } from 'zod';
import {
    type SchemaFormat,
    type FormatterOptions,
    generateTemplateForSchema,
    parseContentForSchema,
} from './schemaFormatters';
import { zodResponseFormat } from 'openai/helpers/zod.mjs';
import {
    ChatCompletionMessageFunctionToolCall,
    type ReasoningEffort,
    type ChatCompletionChunk,
} from 'openai/resources.mjs';
import { CompletionSignal, Message, MessageContent, MessageRole } from './common';
import { ToolCallResult, ToolDefinition, toOpenAITool } from '../tools/types';
import { AgentActionKey, AI_MODEL_CONFIG, AIModelConfig, AIModels, InferenceMetadata, type InferenceRuntimeOverrides } from './config.types';
import { RateLimitService } from '../../services/rate-limit/rateLimits';
import { getUserConfigurableSettings } from '../../config';
import { SecurityError, RateLimitExceededError } from 'shared/types/errors';
import { RateLimitType } from 'worker/services/rate-limit/config';
import { getMaxToolCallingDepth, MAX_LLM_MESSAGES } from '../constants';
import { executeToolCallsWithDependencies } from './toolExecution';
import { CompletionDetector } from './completionDetection';

function optimizeInputs(messages: Message[]): Message[] {
    return messages.map((message) => ({
        ...message,
        content: optimizeMessageContent(message.content),
    }));
}

// Streaming tool-call accumulation helpers 
type ToolCallsArray = NonNullable<NonNullable<ChatCompletionChunk['choices'][number]['delta']>['tool_calls']>;
type ToolCallDelta = ToolCallsArray[number];
type ToolAccumulatorEntry = ChatCompletionMessageFunctionToolCall & { index?: number; __order: number };

function synthIdForIndex(i: number): string {
    return `tool_${Date.now()}_${i}_${Math.random().toString(36).slice(2)}`;
}

function accumulateToolCallDelta(
    byIndex: Map<number, ToolAccumulatorEntry>,
    byId: Map<string, ToolAccumulatorEntry>,
    deltaToolCall: ToolCallDelta,
    orderCounterRef: { value: number }
): void {
    const idx = deltaToolCall.index;
    const idFromDelta = deltaToolCall.id;

    let entry: ToolAccumulatorEntry | undefined;

    // Look up existing entry by id or index
    if (idFromDelta && byId.has(idFromDelta)) {
        entry = byId.get(idFromDelta)!;
    } else if (idx !== undefined && byIndex.has(idx)) {
        entry = byIndex.get(idx)!;
    } else {
        // Create new entry
        const provisionalId = idFromDelta || synthIdForIndex(idx ?? byId.size);
        entry = {
            id: provisionalId,
            type: 'function',
            function: {
                name: '',
                arguments: '',
            },
            __order: orderCounterRef.value++,
            ...(idx !== undefined ? { index: idx } : {}),
        };
        if (idx !== undefined) byIndex.set(idx, entry);
        byId.set(provisionalId, entry);
    }

    // Update id if provided and different
    if (idFromDelta && entry.id !== idFromDelta) {
        byId.delete(entry.id);
        entry.id = idFromDelta;
        byId.set(entry.id, entry);
    }

    // Register index if provided and not yet registered
    if (idx !== undefined && entry.index === undefined) {
        entry.index = idx;
        byIndex.set(idx, entry);
    }

    // Update function name - replace if provided
    if (deltaToolCall.function?.name) {
        entry.function.name = deltaToolCall.function.name;
    }

    // Append arguments - accumulate string chunks
    // CRITICAL FIX: Removed the "isComplete" JSON check. 
    // Always accumulate chunks until the stream segment ends.
    if (deltaToolCall.function?.arguments !== undefined) {
        entry.function.arguments += deltaToolCall.function.arguments;
    }
}

function assembleToolCalls(
    byIndex: Map<number, ToolAccumulatorEntry>,
    byId: Map<string, ToolAccumulatorEntry>
): ChatCompletionMessageFunctionToolCall[] {
    if (byIndex.size > 0) {
        return Array.from(byIndex.values())
            .sort((a, b) => (a.index! - b.index!))
            .map((e) => ({ id: e.id, type: 'function' as const, function: { name: e.function.name, arguments: e.function.arguments } }));
    }
    return Array.from(byId.values())
        .sort((a, b) => a.__order - b.__order)
        .map((e) => ({ id: e.id, type: 'function' as const, function: { name: e.function.name, arguments: e.function.arguments } }));
}

function optimizeMessageContent(content: MessageContent): MessageContent {
    if (!content) return content;
    if (Array.isArray(content)) {
        return content.map((item) =>
            item.type === 'text'
                ? { ...item, text: optimizeTextContent(item.text) }
                : item,
        );
    }
    return optimizeTextContent(content);
}

function optimizeTextContent(content: string): string {
    content = content.replace(/[ \t]+$/gm, '');
    content = content.replace(/\n\s*\n\s*\n\s*\n+/g, '\n\n\n');
    content = content.trim();
    return content;
}

function buildGatewayPathname(cleanPathname: string, providerOverride?: AIGatewayProviders): string {
    return providerOverride ? `${cleanPathname}/${providerOverride}` : `${cleanPathname}/compat`;
}

function constructGatewayUrl(url: URL, providerOverride?: AIGatewayProviders): string {
    const cleanPathname = url.pathname.replace(/\/$/, '');
    url.pathname = buildGatewayPathname(cleanPathname, providerOverride);
    return url.toString();
}

export async function buildGatewayUrl(
	env: Env,
	providerOverride?: AIGatewayProviders,
	gatewayOverride?: { baseUrl: string; token: string },
): Promise<string> {
    if (gatewayOverride?.baseUrl) {
        const url = new URL(gatewayOverride.baseUrl);
        return constructGatewayUrl(url, providerOverride);
    }

    if (env.CLOUDFLARE_AI_GATEWAY_URL && 
        env.CLOUDFLARE_AI_GATEWAY_URL !== 'none' && 
        env.CLOUDFLARE_AI_GATEWAY_URL.trim() !== '') {
        try {
            const url = new URL(env.CLOUDFLARE_AI_GATEWAY_URL);
            if (url.protocol === 'http:' || url.protocol === 'https:') {
                const cleanPathname = url.pathname.replace(/\/$/, '');
                url.pathname = buildGatewayPathname(cleanPathname, providerOverride);
                return url.toString();
            }
        } catch (error) {
            console.warn(`Invalid CLOUDFLARE_AI_GATEWAY_URL provided: ${env.CLOUDFLARE_AI_GATEWAY_URL}. Falling back to AI bindings.`);
        }
    }
    
    const gateway = env.AI.gateway(env.CLOUDFLARE_AI_GATEWAY);
    const baseUrl = providerOverride ? await gateway.getUrl(providerOverride) : `${await gateway.getUrl()}compat`;
    return baseUrl;
}

function isValidApiKey(apiKey: string): boolean {
    if (!apiKey || apiKey.trim() === '') return false;
    if (apiKey.trim().toLowerCase() === 'default' || apiKey.trim().toLowerCase() === 'none' || apiKey.trim().length < 10) return false;
    return true;
}

async function getApiKey(
	provider: string,
	env: Env,
	_userId: string,
	runtimeOverrides?: InferenceRuntimeOverrides,
): Promise<string> {
    const runtimeKey = runtimeOverrides?.userApiKeys?.[provider];
    if (runtimeKey && isValidApiKey(runtimeKey)) {
        return runtimeKey;
    }
    
    const providerKeyString = provider.toUpperCase().replaceAll('-', '_');
    const envKey = `${providerKeyString}_API_KEY` as keyof Env;
    let apiKey: string = env[envKey] as string;
    
    if (!isValidApiKey(apiKey)) {
        apiKey = runtimeOverrides?.aiGatewayOverride?.token ?? env.CLOUDFLARE_AI_GATEWAY_TOKEN;
    }
    return apiKey;
}

export async function getConfigurationForModel(
    modelConfig: AIModelConfig,
    env: Env, 
    userId: string,
    runtimeOverrides?: InferenceRuntimeOverrides,
): Promise<{
    baseURL: string,
    apiKey: string,
    defaultHeaders?: Record<string, string>,
}> {
    let providerForcedOverride: AIGatewayProviders | undefined;
    if (modelConfig.directOverride) {
        switch(modelConfig.provider) {
            case 'openrouter': return { baseURL: 'https://openrouter.ai/api/v1', apiKey: env.OPENROUTER_API_KEY };
            case 'google-ai-studio': return { baseURL: 'https://generativelanguage.googleapis.com/v1beta/openai/', apiKey: env.GOOGLE_AI_STUDIO_API_KEY };
            case 'anthropic': return { baseURL: 'https://api.anthropic.com/v1/', apiKey: env.ANTHROPIC_API_KEY };
            default:
                providerForcedOverride = modelConfig.provider as AIGatewayProviders;
                break;
        }
    }

    const gatewayOverride = runtimeOverrides?.aiGatewayOverride;
    const baseURL = await buildGatewayUrl(env, providerForcedOverride, gatewayOverride);
    const gatewayToken = gatewayOverride?.token ?? env.CLOUDFLARE_AI_GATEWAY_TOKEN;
    const apiKey = await getApiKey(modelConfig.provider, env, userId, runtimeOverrides);

    const defaultHeaders = gatewayToken && apiKey !== gatewayToken ? {
        'cf-aig-authorization': `Bearer ${gatewayToken}`,
    } : undefined;
    return { baseURL, apiKey, defaultHeaders };
}

type InferArgsBase = {
    env: Env;
    metadata: InferenceMetadata;
    actionKey: AgentActionKey  | 'testModelConfig';
    messages: Message[];
    maxTokens?: number;
    modelName: AIModels | string;
    reasoning_effort?: ReasoningEffort;
    temperature?: number;
    frequency_penalty?: number;
    stream?: {
        chunk_size: number;
        onChunk: (chunk: string) => void;
    };
    tools?: ToolDefinition<any, any>[];
    providerOverride?: 'cloudflare' | 'direct';
    runtimeOverrides?: InferenceRuntimeOverrides;
    abortSignal?: AbortSignal;
    onAssistantMessage?: (message: Message) => Promise<void>;
    completionConfig?: CompletionConfig;
};

type InferArgsStructured = InferArgsBase & {
    schema: z.AnyZodObject;
    schemaName: string;
};

type InferWithCustomFormatArgs = InferArgsStructured & {
    format?: SchemaFormat;
    formatOptions?: FormatterOptions;
};

export interface ToolCallContext {
    messages: Message[];
    depth: number;
    completionSignal?: CompletionSignal;
    warningInjected?: boolean;
}

export interface CompletionConfig {
    detector?: CompletionDetector;
    operationalMode?: 'initial' | 'followup';
    allowWarningInjection?: boolean;
}

export function serializeCallChain(context: ToolCallContext, finalResponse: string): string {
    let transcript = '**Request terminated by user, partial response transcript (last 5 messages):**\n\n<call_chain_transcript>';
    for (const message of context.messages.slice(-5)) {
        let content = message.content;
        if (message.role === 'tool' || message.role === 'function') {
            content = (content || '').slice(0, 100);
        }
        transcript += `<message role="${message.role}">${content}</message>`;
    }
    transcript += `<final_response>${finalResponse || '**cancelled**'}</final_response>`;
    transcript += '</call_chain_transcript>';
    return transcript;
}

export class InferError extends Error {
    constructor(
        message: string,
        public response: string,
        public toolCallContext?: ToolCallContext
    ) {
        super(message);
        this.name = 'InferError';
    }

    partialResponseTranscript(): string {
        if (!this.toolCallContext) {
            return this.response;
        }
        return serializeCallChain(this.toolCallContext, this.response);
    }

    partialResponse(): InferResponseString {
        return {
            string: this.response,
            toolCallContext: this.toolCallContext
        };
    }
}

export class AbortError extends InferError {
    constructor(response: string, toolCallContext?: ToolCallContext) {
        super(response, response, toolCallContext);
        this.name = 'AbortError';
    }
}

const claude_thinking_budget_tokens = {
    medium: 8000,
    high: 16000,
    low: 4000,
    minimal: 1000,
};

export type InferResponseObject<OutputSchema extends z.AnyZodObject> = {
    object: z.infer<OutputSchema>;
    toolCallContext?: ToolCallContext;
};

export type InferResponseString = {
    string: string;
    toolCallContext?: ToolCallContext;
};

async function executeToolCalls(openAiToolCalls: ChatCompletionMessageFunctionToolCall[], originalDefinitions: ToolDefinition[]): Promise<ToolCallResult[]> {
    return executeToolCallsWithDependencies(openAiToolCalls, originalDefinitions);
}

function updateToolCallContext(
    toolCallContext: ToolCallContext | undefined,
    assistantMessage: Message,
    executedToolCalls: ToolCallResult[],
    completionDetector?: CompletionDetector
) {
    const newMessages = [
        ...(toolCallContext?.messages || []),
        assistantMessage,
        ...executedToolCalls
            .filter(result => result.name && result.name.trim() !== '')
            .map((result, _) => ({
                role: "tool" as MessageRole,
                content: result.result ? JSON.stringify(result.result) : 'done',
                name: result.name,
                tool_call_id: result.id,
            })),
        ];

    const newDepth = (toolCallContext?.depth ?? 0) + 1;
    let completionSignal = toolCallContext?.completionSignal;
    if (completionDetector && !completionSignal) {
        completionSignal = completionDetector.detectCompletion(executedToolCalls);
    }

    const newToolCallContext: ToolCallContext = {
        messages: newMessages,
        depth: newDepth,
        completionSignal,
        warningInjected: toolCallContext?.warningInjected || false
    };
    return newToolCallContext;
}

export function infer<OutputSchema extends z.AnyZodObject>(args: InferArgsStructured, toolCallContext?: ToolCallContext): Promise<InferResponseObject<OutputSchema>>;
export function infer(args: InferArgsBase, toolCallContext?: ToolCallContext): Promise<InferResponseString>;
export function infer<OutputSchema extends z.AnyZodObject>(args: InferWithCustomFormatArgs, toolCallContext?: ToolCallContext): Promise<InferResponseObject<OutputSchema>>;

export async function infer<OutputSchema extends z.AnyZodObject>({
    env,
    metadata,
    messages,
    schema,
    schemaName,
    actionKey,
    format,
    formatOptions,
    modelName,
    reasoning_effort,
    temperature,
    frequency_penalty,
    maxTokens,
    stream,
    tools,
    runtimeOverrides,
    abortSignal,
    onAssistantMessage,
    completionConfig,
}: InferArgsBase & {
    schema?: OutputSchema;
    schemaName?: string;
    format?: SchemaFormat;
    formatOptions?: FormatterOptions;
}, toolCallContext?: ToolCallContext): Promise<InferResponseObject<OutputSchema> | InferResponseString> {
    if (messages.length > MAX_LLM_MESSAGES) {
        throw new RateLimitExceededError(`Message limit exceeded: ${messages.length} messages (max: ${MAX_LLM_MESSAGES}).`, RateLimitType.LLM_CALLS);
    }
    
    const currentDepth = toolCallContext?.depth ?? 0;
    if (currentDepth >= getMaxToolCallingDepth(actionKey)) {
        if (schema) {
            throw new AbortError(`Maximum tool calling depth (${getMaxToolCallingDepth(actionKey)}) exceeded.`, toolCallContext);
        }
        return { string: `[System: Maximum tool calling depth reached.]`, toolCallContext };
    }
    
    try {
        const userConfig = await getUserConfigurableSettings(env, metadata.userId)
        await RateLimitService.enforceLLMCallsRateLimit(env, userConfig.security.rateLimit, metadata.userId, modelName)
        const modelConfig = AI_MODEL_CONFIG[modelName as AIModels];

        const { apiKey, baseURL, defaultHeaders } = await getConfigurationForModel(
            modelConfig,
            env,
            metadata.userId,
            runtimeOverrides,
        );

        modelName = modelName.replace(/\[.*?\]/, '');
        const client = new OpenAI({ apiKey, baseURL: baseURL, defaultHeaders });
        
        const schemaObj = schema && schemaName && !format ? { response_format: zodResponseFormat(schema, schemaName) } : {};
        const extraBody = modelName.includes('claude') ? {
                    extra_body: {
                        thinking: {
                            type: 'enabled',
                            budget_tokens: claude_thinking_budget_tokens[reasoning_effort ?? 'medium'],
                        },
                    },
                } : {};

        const optimizedMessages = optimizeInputs(messages);
        let messagesToPass = [...optimizedMessages];
        
        if (toolCallContext && toolCallContext.messages) {
            const ctxMessages = toolCallContext.messages;
            let validToolCallIds = new Set<string>();

            let filtered = ctxMessages.filter(msg => {
                if (msg.role === 'assistant' && msg.tool_calls) {
                    validToolCallIds = new Set(msg.tool_calls.map(tc => tc.id));
                    return true;
                }
                if (msg.role === 'tool') {
                    if (!msg.name?.trim() || !msg.tool_call_id || !validToolCallIds.has(msg.tool_call_id)) return false;
                }
                return true;
            });

            messagesToPass.push(...filtered);
        }

        if (format) {
            if (!schema || !schemaName) throw new Error('Schema and schemaName are required when using a custom format');
            const formatInstructions = generateTemplateForSchema(schema, format, formatOptions);
            const lastMessage = messagesToPass[messagesToPass.length - 1];

            if (typeof lastMessage.content === 'string') {
                messagesToPass = [...messagesToPass.slice(0, -1), { role: lastMessage.role, content: `${lastMessage.content}\n\n${formatInstructions}` }];
            } else if (Array.isArray(lastMessage.content)) {
                const updatedContent = lastMessage.content.map((item) => {
                    if (item.type === 'text') {
                        return { ...item, text: `${item.text}\n\n${formatInstructions}` };
                    }
                    return item;
                });
                messagesToPass = [...messagesToPass.slice(0, -1), { role: lastMessage.role, content: updatedContent }];
            }
        }

        const toolsOpts = tools ? {
            tools: tools.map(t => toOpenAITool(t)),
            tool_choice: 'auto' as const
        } : {};

        let response: OpenAI.ChatCompletion | OpenAI.ChatCompletionChunk | Stream<OpenAI.ChatCompletionChunk>;
        try {
            response = await client.chat.completions.create({
                ...schemaObj,
                ...extraBody,
                ...toolsOpts,
                model: modelName,
                messages: messagesToPass as OpenAI.ChatCompletionMessageParam[],
                max_completion_tokens: maxTokens || 150000,
                stream: stream ? true : false,
                reasoning_effort: modelConfig.nonReasoning ? undefined : reasoning_effort,
                temperature,
                frequency_penalty,
            }, {
                signal: abortSignal,
                headers: {
                    "cf-aig-metadata": JSON.stringify({
                        chatId: metadata.agentId,
                        userId: metadata.userId,
                        schemaName,
                        actionKey,
                    })
                }
            });
        } catch (error) {
            if (error instanceof Error && (error.name === 'AbortError' || error.message?.includes('aborted') || error.message?.includes('abort'))) {
                throw new AbortError('**User cancelled inference**', toolCallContext);
            }
            throw error;
        }

        let toolCalls: ChatCompletionMessageFunctionToolCall[] = [];
        let content = '';

        if (stream) {
            if (response instanceof Stream) {
                let streamIndex = 0;
                const byIndex = new Map<number, ToolAccumulatorEntry>();
                const byId = new Map<string, ToolAccumulatorEntry>();
                const orderCounterRef = { value: 0 };
                
                for await (const event of response) {
                    const delta = (event as ChatCompletionChunk).choices[0]?.delta;
                    
                    if (delta?.tool_calls) {
                        try {
                            for (const deltaToolCall of delta.tool_calls as ToolCallsArray) {
                                accumulateToolCallDelta(byIndex, byId, deltaToolCall, orderCounterRef);
                            }
                        } catch (error) {
                            console.error('Error processing tool calls in streaming:', error);
                        }
                    }
                    
                    content += delta?.content || '';
                    const slice = content.slice(streamIndex);
                    const finishReason = (event as ChatCompletionChunk).choices[0]?.finish_reason;
                    if (slice.length >= stream.chunk_size || finishReason != null) {
                        stream.onChunk(slice);
                        streamIndex += slice.length;
                    }
                }
                
                const assembled = assembleToolCalls(byIndex, byId);
                toolCalls = assembled.filter(tc => tc.function.name && tc.function.name.trim() !== '');
            } else {
                const completion = response as OpenAI.ChatCompletion;
                const message = completion.choices[0]?.message;
                if (message) {
                    content = message.content || '';
                    toolCalls = (message.tool_calls as ChatCompletionMessageFunctionToolCall[]) || [];
                }
            }
        } else {
            const completion = response as OpenAI.ChatCompletion;
            content = completion.choices[0]?.message?.content || '';
            const allToolCalls = (completion.choices[0]?.message?.tool_calls as ChatCompletionMessageFunctionToolCall[] || []);
            toolCalls = allToolCalls.filter(tc => tc.function.name && tc.function.name.trim() !== '');
        }

        const assistantMessage = { role: "assistant" as MessageRole, content, tool_calls: toolCalls };
        if (onAssistantMessage) {
            await onAssistantMessage(assistantMessage);
        }

        if (!content && !stream && !toolCalls.length) {
            return { string: "", toolCallContext };
        }

        let executedToolCalls: ToolCallResult[] = [];
        if (tools) {
            try {
                executedToolCalls = await executeToolCalls(toolCalls, tools);
            } catch (error) {
                if (error instanceof AbortError) {
                    const newToolCallContext = updateToolCallContext(toolCallContext, assistantMessage, executedToolCalls, completionConfig?.detector);
                    return { string: content, toolCallContext: newToolCallContext };
                }
            }
        }

        if (executedToolCalls.length) {
            const newToolCallContext = updateToolCallContext(toolCallContext, assistantMessage, executedToolCalls, completionConfig?.detector);

            if (newToolCallContext.completionSignal?.signaled) {
                if (schema && schemaName) {
                    throw new AbortError(
                        `Completion signaled: ${newToolCallContext.completionSignal.summary || 'Task complete'}`,
                        newToolCallContext
                    );
                }
                return {
                    string: content || newToolCallContext.completionSignal.summary || 'Task complete',
                    toolCallContext: newToolCallContext
                };
            }

            const executedCallsWithResults = executedToolCalls.filter(result =>
                result.result !== undefined &&
                !(completionConfig?.detector?.isCompletionTool(result.name))
            );
            
            if (executedCallsWithResults.length) {
                if (schema && schemaName) {
                    return await infer<OutputSchema>({
                        env, metadata, messages, schema, schemaName, format, formatOptions,
                        actionKey, modelName, maxTokens, stream, tools, reasoning_effort,
                        temperature, frequency_penalty, abortSignal, onAssistantMessage, completionConfig,
                    }, newToolCallContext);
                } else {
                    return await infer({
                        env, metadata, messages, modelName, maxTokens, actionKey, stream,
                        tools, reasoning_effort, temperature, frequency_penalty, abortSignal,
                        onAssistantMessage, completionConfig,
                    }, newToolCallContext);
                }
            } else {
                return { string: content, toolCallContext: newToolCallContext };
            }
        }

        if (!schema) {
            return { string: content, toolCallContext };
        }

        try {
            const parsedContent = format
                ? parseContentForSchema(content, format, schema, formatOptions)
                : JSON.parse(content);

            const result = schema.safeParse(parsedContent);
            if (!result.success) {
                throw new Error(`Failed to validate AI response against schema: ${result.error.message}`);
            }
            return { object: result.data, toolCallContext };
        } catch (parseError) {
            throw new InferError('Failed to parse response', content, toolCallContext);
        }
    } catch (error) {
        if (error instanceof RateLimitExceededError || error instanceof SecurityError) {
            throw error;
        }
        throw error;
    }
}
