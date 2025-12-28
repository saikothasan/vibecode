import { Agent, AgentContext, ConnectionContext } from "agents";
import { AgentInitArgs, AgentSummary, DeployOptions, DeployResult, ExportOptions, ExportResult, DeploymentTarget, BehaviorType } from "./types";
import { AgenticState, AgentState, BaseProjectState, CurrentDevState, MAX_PHASES, PhasicState } from "./state";
import { Blueprint } from "../schemas";
import { BaseCodingBehavior } from "./behaviors/base";
import { createObjectLogger, StructuredLogger } from '../../logger';
import { InferenceMetadata } from "../inferutils/config.types";
import { getMimeType } from 'hono/utils/mime';
import { normalizePath, isPathSafe } from '../../utils/pathUtils';
import { FileManager } from '../services/implementations/FileManager';
import { DeploymentManager } from '../services/implementations/DeploymentManager';
import { GitVersionControl } from '../git';
import { StateManager } from '../services/implementations/StateManager';
import { PhasicCodingBehavior } from './behaviors/phasic';
import { AgenticCodingBehavior } from './behaviors/agentic';
import { SqlExecutor } from '../git';
import { AgentInfrastructure } from "./AgentCore";
import { ProjectType } from './types';
import { Connection } from 'agents';
import { handleWebSocketMessage, handleWebSocketClose, broadcastToConnections, sendToConnection } from './websocket';
import { WebSocketMessageData, WebSocketMessageType } from "worker/api/websocketTypes";
import { PreviewType, TemplateDetails } from "worker/services/sandbox/sandboxTypes";
import { WebSocketMessageResponses } from "../constants";
import { AppService, ModelConfigService } from "worker/database";
import { ConversationMessage, ConversationState } from "../inferutils/common";
import { ImageAttachment } from "worker/types/image-attachment";
import { RateLimitExceededError } from "shared/types/errors";
import { ProjectObjective } from "./objectives/base";
import { FileOutputType } from "../schemas";
import { SecretsClient, type UserSecretsStoreStub } from '../../services/secrets/SecretsClient';
import { StateMigration } from './stateMigration';

const DEFAULT_CONVERSATION_SESSION_ID = 'default';

interface AgentBootstrapProps {
    behaviorType?: BehaviorType;
    projectType?: ProjectType;
}

export class CodeGeneratorAgent extends Agent<Env, AgentState> implements AgentInfrastructure<AgentState> {
    public _logger: StructuredLogger | undefined;
    private behavior!: BaseCodingBehavior<AgentState>;
    private objective!: ProjectObjective<BaseProjectState>;
    private secretsClient: SecretsClient | null = null;
    protected static readonly PROJECT_NAME_PREFIX_MAX_LENGTH = 20;
    // Services
    readonly fileManager: FileManager;
    readonly deploymentManager: DeploymentManager;
    readonly git: GitVersionControl;
    
    // Redeclare as public to satisfy AgentInfrastructure interface
    declare public readonly env: Env;
    declare public readonly sql: SqlExecutor;
    
    // ==========================================
    // Initialization
    // ==========================================
    
    initialState = {
        behaviorType: 'unknown' as BehaviorType,
        projectType: 'unknown' as ProjectType,
        projectName: "",
        query: "",
        sessionId: '',
        hostname: '',
        blueprint: {} as unknown as Blueprint,
        templateName: '',
        generatedFilesMap: {},
        conversationMessages: [],
        metadata: {} as InferenceMetadata,
        shouldBeGenerating: false,
        sandboxInstanceId: undefined,
        commandsHistory: [],
        lastPackageJson: '',
        pendingUserInputs: [],
        projectUpdatesAccumulator: [],
        lastDeepDebugTranscript: null,
        mvpGenerated: false,
        reviewingInitiated: false,
        generatedPhases: [],
        currentDevState: CurrentDevState.IDLE,
        phasesCounter: MAX_PHASES,
    } as AgentState;

    constructor(ctx: AgentContext, env: Env) {
        super(ctx, env);
        
        try {
            void this.sql`CREATE TABLE IF NOT EXISTS full_conversations (id TEXT PRIMARY KEY, messages TEXT)`;
            void this.sql`CREATE TABLE IF NOT EXISTS compact_conversations (id TEXT PRIMARY KEY, messages TEXT)`;
        } catch (e) {
            console.error("Failed to initialize SQLite tables", e);
        }

        // Create StateManager
        const stateManager = new StateManager(
            () => this.state,
            (s) => this.setState(s)
        );
        
        this.git = new GitVersionControl(this.sql.bind(this));
        this.fileManager = new FileManager(
            stateManager,
            () => this.behavior?.getTemplateDetails?.() || null,
            this.git
        );
        this.deploymentManager = new DeploymentManager(
            {
                stateManager,
                fileManager: this.fileManager,
                getLogger: () => this.logger(),
                env: this.env
            },
            10, // MAX_COMMANDS_HISTORY
        );
    }
    private createObjective(projectType: ProjectType): ProjectObjective<BaseProjectState> {
        return new ProjectObjective(this as AgentInfrastructure<BaseProjectState>, projectType);
    }

    /**
     * Initialize the agent with project blueprint and template
     * Only called once in an app's lifecycle
     */
    async initialize(
        initArgs: AgentInitArgs<AgentState>,
        ..._args: unknown[]
    ): Promise<AgentState> {
        const { inferenceContext } = initArgs;
        const sandboxSessionId = DeploymentManager.generateNewSessionId();
        this.initLogger(inferenceContext.metadata.agentId, inferenceContext.metadata.userId, sandboxSessionId);

        try {
            // Infrastructure setup
            await this.gitInit();
            
            // Let behavior handle all state initialization (blueprint, projectName, etc.)
            await this.behavior.initialize({
                ...initArgs,
                sandboxSessionId // Pass generated session ID to behavior
            });
            
            await this.saveToDatabase();
        } catch (error) {
            this.logger().error("Agent initialization failed", error);
            throw error;
        }
        
        return this.state;
    }
    
    async isInitialized() {
        return this.getAgentId() ? true : false
    }

    /**
     * Called evertime when agent is started or re-started
     * @param props - Optional props
     */
    async onStart(props?: Record<string, unknown> | undefined): Promise<void> {
        try {
            // Run common migration FIRST, before any state access
            const migratedState = StateMigration.migrateCommon(this.state);
            if (migratedState) {
                this.setState(migratedState);
            }

            this.logger().info(`Agent ${this.getAgentId()} session: ${this.state.sessionId} onStart`, { props });

            this.logger().info('Bootstrapping CodeGeneratorAgent', { props });
            const agentProps = props as AgentBootstrapProps;
            const behaviorType = agentProps?.behaviorType ?? this.state.behaviorType ?? 'phasic';
            const projectType = agentProps?.projectType ?? this.state.projectType ?? 'app';

            if (behaviorType === 'phasic') {
                this.behavior = new PhasicCodingBehavior(this as AgentInfrastructure<PhasicState>, projectType);
            } else {
                this.behavior = new AgenticCodingBehavior(this as AgentInfrastructure<AgenticState>, projectType);
            }
            
            // Create objective based on project type
            this.objective = this.createObjective(projectType);

            this.behavior.onStart(props);

            // Ignore if agent not initialized
            if (!this.state.query) {
                this.logger().warn(`Agent ${this.getAgentId()} session: ${this.state.sessionId} onStart ignored, agent not initialized`);
                return;
            }

            // Ensure state is migrated for any previous versions
            this.behavior.migrateStateIfNeeded();
            
            // Check if this is a read-only operation
            const readOnlyMode = props?.readOnlyMode === true;
            
            if (readOnlyMode) {
                this.logger().info(`Agent ${this.getAgentId()} starting in READ-ONLY mode - skipping expensive initialization`);
                return;
            }
            
            // Just in case
            await this.gitInit();
            
            await this.behavior.ensureTemplateDetails();
            this.logger().info(`Agent ${this.getAgentId()} session: ${this.state.sessionId} onStart processed successfully`);

            // Load the latest user configs
            try {
                const modelConfigService = new ModelConfigService(this.env);
                const userConfigsRecord = await modelConfigService.getUserModelConfigs(this.state.metadata.userId);
                this.behavior.setUserModelConfigs(userConfigsRecord);
                this.logger().info(`Agent ${this.getAgentId()} session: ${this.state.sessionId} onStart: User configs loaded successfully`, {userConfigsRecord});
            } catch (configError) {
                this.logger().warn("Failed to load user model configs, using defaults", configError);
            }
        } catch (error) {
            this.logger().error("Fatal error in onStart", error);
            // Don't rethrow, let the agent stay alive so we can debug via WS
        }
    }
    
    onConnect(connection: Connection, ctx: ConnectionContext) {
        this.logger().info(`Agent connected for agent ${this.getAgentId()}`, { connection, ctx });
        let previewUrl = '';
        try {
            if (this.behavior && this.behavior.getTemplateDetails() && this.behavior.getTemplateDetails().renderMode === 'browser') {
                previewUrl = this.behavior.getBrowserPreviewURL();
            }
        } catch (error) {
            this.logger().error('Error getting preview URL:', error);
        }
        sendToConnection(connection, WebSocketMessageResponses.AGENT_CONNECTED, {
            state: this.state,
            templateDetails: this.behavior?.getTemplateDetails?.() || null,
            previewUrl: previewUrl
        });
    }

    private initLogger(agentId: string, userId: string, sessionId?: string) {
        this._logger = createObjectLogger(this, 'CodeGeneratorAgent');
        this._logger.setObjectId(agentId);
        this._logger.setFields({
            agentId,
            userId,
            projectType: this.state.projectType,
            behaviorType: this.state.behaviorType
        });
        if (sessionId) {
            this._logger.setField('sessionId', sessionId);
        }
        return this._logger;
    }
    
    // ==========================================
    // Utilities
    // ==========================================

    logger(): StructuredLogger {
        if (!this._logger) {
            // Fallback logger if state isn't ready
            const agentId = this.state.metadata?.agentId || 'unknown_agent';
            const userId = this.state.metadata?.userId || 'unknown_user';
            this._logger = this.initLogger(agentId, userId, this.state.sessionId);
        }
        return this._logger;
    }

    getAgentId() {
        return this.state.metadata?.agentId;
    }
    
    getWebSockets(): WebSocket[] {
        return this.ctx.getWebSockets();
    }

    handleVaultUnlocked(): void {
        this.secretsClient?.notifyUnlocked();
        this.logger().info('Vault unlocked notification received', {});
    }

    handleVaultLocked(): void {
        this.secretsClient?.notifyUnlockFailed('Vault locked');
        this.secretsClient = null;
        this.logger().info('Vault locked', {});
    }

    private getSecretsClient(): SecretsClient {
        if (!this.secretsClient) {
            const userId = this.state.metadata.userId;
            const stub = this.env.UserSecretsStore.get(
                this.env.UserSecretsStore.idFromName(userId)
            ) as unknown as UserSecretsStoreStub;

            this.secretsClient = new SecretsClient(stub, (type, data) => {
                if (type === 'vault_required') {
                    const vaultData = data as { reason: string; provider?: string; envVarName?: string; secretId?: string };
                    broadcastToConnections(this, 'vault_required', vaultData);
                }
            });
        }

        return this.secretsClient;
    }

    async getDecryptedSecret(query: { provider?: string; envVarName?: string; secretId?: string }): Promise<string | null> {
        try {
            return await this.getSecretsClient().get(query);
        } catch (error) {
            this.logger().info('Secret request failed', { query, error: String(error) });
            return null;
        }
    }

    /**
     * Get the project objective (defines what is being built)
     */
    getObjective(): ProjectObjective<BaseProjectState> {
        return this.objective;
    }
    
    /**
     * Get the behavior (defines how code is generated)
     */
    getBehavior(): BaseCodingBehavior<AgentState> {
        return this.behavior;
    }

    async getFullState(): Promise<AgentState> {
        if (!this.behavior) return this.state;
        return await this.behavior.getFullState();
    }

    async getSummary(): Promise<AgentSummary> {
        if (!this.behavior) {
            return {
                agentId: this.getAgentId(),
                title: 'Initializing...',
                description: 'Please wait',
                status: 'creating',
                updatedAt: new Date().toISOString()
            };
        }
        return this.behavior.getSummary();
    }

    getPreviewUrlCache(): string {
        return ''; // Unimplemented
    }

    deployToSandbox(
        files: FileOutputType[] = [],
        redeploy: boolean = false,
        commitMessage?: string,
        clearLogs: boolean = false
    ): Promise<PreviewType | null> {
        return this.behavior.deployToSandbox(files, redeploy, commitMessage, clearLogs);
    }

    deployToCloudflare(target?: DeploymentTarget): Promise<{ deploymentUrl?: string; workersUrl?: string } | null> {
        return this.behavior.deployToCloudflare(target);
    }

    deployProject(options?: DeployOptions): Promise<DeployResult> {
        return this.objective.deploy(options);
    }

    exportProject(options: ExportOptions): Promise<ExportResult> {
        return this.objective.export(options);
    }

    importTemplate(templateName: string): Promise<{ templateName: string; filesImported: number }> {
        return this.behavior.importTemplate(templateName);
    }
    
    protected async saveToDatabase() {
        this.logger().info(`Saving agent ${this.getAgentId()} to database`);
        // Save the app to database (authenticated users only)
        try {
            const appService = new AppService(this.env);
            await appService.createApp({
                id: this.state.metadata.agentId,
                userId: this.state.metadata.userId,
                sessionToken: null,
                title: this.state.blueprint?.title || this.state.query?.substring(0, 100) || 'Untitled App',
                description: this.state.blueprint?.description || '',
                originalPrompt: this.state.query || '',
                finalPrompt: this.state.query || '',
                framework: this.state.blueprint?.frameworks?.join(',') || 'react',
                visibility: 'private',
                status: 'generating',
                createdAt: new Date(),
                updatedAt: new Date()
                });
            this.logger().info(`App saved successfully to database for agent ${this.state.metadata.agentId}`);
        } catch (error) {
            this.logger().error(`Failed to save app to database`, error);
            // Don't throw here, as we still want the agent to work in memory
        }
    }

    // ==========================================
    // Conversation Management
    // ==========================================

    getConversationState(id: string = DEFAULT_CONVERSATION_SESSION_ID): ConversationState {
        let fullHistory: ConversationMessage[] = [];
        let runningHistory: ConversationMessage[] = [];

        try {
            const rows = this.sql<{ messages: string, id: string }>`SELECT * FROM full_conversations WHERE id = ${id}`;
            if (rows.length > 0 && rows[0].messages) {
                fullHistory = JSON.parse(rows[0].messages) as ConversationMessage[];
            }
            
            const compactRows = this.sql<{ messages: string, id: string }>`SELECT * FROM compact_conversations WHERE id = ${id}`;
            if (compactRows.length > 0 && compactRows[0].messages) {
                runningHistory = JSON.parse(compactRows[0].messages) as ConversationMessage[];
            }
        } catch (e) {
            this.logger().warn('Failed to load conversation history from SQL', e);
        }

        if (runningHistory.length === 0) {
            runningHistory = fullHistory;
        }

        // Remove duplicates
        const deduplicateMessages = (messages: ConversationMessage[]): ConversationMessage[] => {
            const seen = new Set<string>();
            return messages.filter(msg => {
                const key = `${msg.conversationId}-${msg.role}-${msg.tool_call_id || ''}`;
                if (seen.has(key)) return false;
                seen.add(key);
                return true;
            });
        };

        runningHistory = deduplicateMessages(runningHistory);
        fullHistory = deduplicateMessages(fullHistory);
        
        return { id: id, runningHistory, fullHistory };
    }

    setConversationState(conversations: ConversationState) {
        const serializedFull = JSON.stringify(conversations.fullHistory);
        const serializedCompact = JSON.stringify(conversations.runningHistory);
        try {
            this.logger().info(`Saving conversation state ${conversations.id}`);
            void this.sql`INSERT OR REPLACE INTO compact_conversations (id, messages) VALUES (${conversations.id}, ${serializedCompact})`;
            void this.sql`INSERT OR REPLACE INTO full_conversations (id, messages) VALUES (${conversations.id}, ${serializedFull})`;
        } catch (error) {
            this.logger().error(`Failed to save conversation state ${conversations.id}`, error);
        }
    }

    addConversationMessage(message: ConversationMessage) {
        const conversationState = this.getConversationState();
        
        const updateHistory = (history: ConversationMessage[]) => {
            const index = history.findIndex(msg => msg.conversationId === message.conversationId);
            if (index !== -1) {
                history[index] = message;
            } else {
                history.push(message);
            }
            return history;
        };

        conversationState.runningHistory = updateHistory(conversationState.runningHistory);
        conversationState.fullHistory = updateHistory(conversationState.fullHistory);
        
        this.setConversationState(conversationState);
    }
    
    public clearConversation(): void {
        try {
            this.logger().info('Clearing conversation history');
            void this.sql`DELETE FROM full_conversations WHERE id = ${DEFAULT_CONVERSATION_SESSION_ID}`;
            void this.sql`DELETE FROM compact_conversations WHERE id = ${DEFAULT_CONVERSATION_SESSION_ID}`;
            
            this.broadcast(WebSocketMessageResponses.CONVERSATION_CLEARED, {
                message: 'Conversation history cleared',
            });
        } catch (error) {
            this.logger().error('Error clearing conversation history:', error);
            this.broadcastError('Failed to clear conversation history', error);
        }
    }

    async handleUserInput(userMessage: string, images?: ImageAttachment[]): Promise<void> {
        try {
            this.logger().info('Processing user input message');
            if (!this.behavior) {
                throw new Error("Agent behavior not initialized");
            }

            await this.behavior.handleUserInput(userMessage, images);
            
            if (!this.behavior.isCodeGenerating()) {
                this.logger().info('User input during IDLE state, starting generation');
                this.behavior.generateAllFiles().catch(error => {
                    this.logger().error('Error starting generation from user input:', error);
                });
            }

        } catch (error) {
            if (error instanceof RateLimitExceededError) {
                this.broadcast(WebSocketMessageResponses.RATE_LIMIT_ERROR, { error });
                return;
            }
            this.broadcastError('Error processing user input', error);
        }
    }

    async onMessage(connection: Connection, message: string): Promise<void> {
        handleWebSocketMessage(this, connection, message);
    }
    
    async onClose(connection: Connection): Promise<void> {
        handleWebSocketClose(this, connection);
    }
    
    public broadcast<T extends WebSocketMessageType>(
        type: T, 
        data?: WebSocketMessageData<T>
    ): void {
        broadcastToConnections(this, type, data || {} as WebSocketMessageData<T>);
    }

    protected broadcastError(context: string, error: unknown): void {
        const errorMessage = error instanceof Error ? error.message : String(error);
        this.logger().error(`${context}:`, error);
        this.broadcast(WebSocketMessageResponses.ERROR, {
            error: `${context}: ${errorMessage}`
        });
    }

    protected async gitInit() {
        try {
            await this.git.init();
            const head = await this.git.getHead();
            
            if (!head) {
                const generatedFiles = this.fileManager.getGeneratedFiles();
                if (generatedFiles.length > 0) {
                    await this.git.commit(generatedFiles, "Initial commit");
                }
            }
        } catch (error) {
            this.logger().error("Error during git init:", error);
        }
    }

    async exportGitObjects(): Promise<{
        gitObjects: Array<{ path: string; data: Uint8Array }>;
        query: string;
        hasCommits: boolean;
        templateDetails: TemplateDetails | null;
    }> {
        try {
            const gitObjects = this.git.fs.exportGitObjects();
            await this.gitInit();
            
            if (this.behavior) {
                await this.behavior.ensureTemplateDetails();
            }

            const templateDetails = this.behavior?.getTemplateDetails() || null;
            
            return {
                gitObjects,
                query: this.state.query || 'N/A',
                hasCommits: gitObjects.length > 0,
                templateDetails
            };
        } catch (error) {
            this.logger().error('exportGitObjects failed', error);
            throw error;
        }
    }

    async handleBrowserFileServing(request: Request): Promise<Response> {
        const url = new URL(request.url);

        if (request.method === 'OPTIONS') {
            return new Response(null, {
                status: 204,
                headers: {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'GET, OPTIONS',
                    'Access-Control-Allow-Headers': '*',
                    'Access-Control-Max-Age': '86400'
                }
            });
        }

        const subdomain = url.hostname.split('.')[0];
        if (!subdomain.startsWith('b-')) return new Response('Invalid request', { status: 400 });

        const withoutPrefix = subdomain.substring(2);
        const lastHyphenIndex = withoutPrefix.lastIndexOf('-');
        if (lastHyphenIndex === -1) return new Response('Invalid request', { status: 400 });

        const providedToken = withoutPrefix.substring(lastHyphenIndex + 1);
        const filePath = url.pathname === '/' || url.pathname === '' ? 'public/index.html' : url.pathname.replace(/^\//, '');

        const storedToken = this.state.fileServingToken?.token;
        if (!storedToken || providedToken !== storedToken.toLowerCase()) {
            return new Response('Unauthorized', { status: 403 });
        }

        if (!isPathSafe(filePath)) return new Response('Invalid path', { status: 400 });
        
        const normalized = normalizePath(filePath);
        let file = this.fileManager.getFile(normalized);
        if (!file && !normalized.startsWith('public/')) {
            file = this.fileManager.getFile(`public/${normalized}`);
        }

        if (!file) return new Response('File not found', { status: 404 });

        const contentType = getMimeType(normalized) || 'application/octet-stream';
        let content = file.fileContents;

        if (normalized.endsWith('.html') || contentType.includes('text/html')) {
            const baseTag = `<base href="/">`;
            if (content.includes('<head>')) {
                content = content.replace(/<head>/i, `<head>\n  ${baseTag}`);
            } else {
                content = baseTag + '\n' + content;
            }
        }

        return new Response(content, {
            status: 200,
            headers: {
                'Content-Type': contentType,
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Access-Control-Allow-Origin': '*',
                'X-Sandbox-Type': 'browser-native'
            }
        });
    }

    setGitHubToken(token: string, username: string, ttl: number = 3600000): void {
        this.objective.setGitHubToken(token, username, ttl);
    }

    getGitHubToken(): { token: string; username: string } | null {
        return this.objective.getGitHubToken();
    }

    clearGitHubToken(): void {
        this.objective.clearGitHubToken();
    }
}
