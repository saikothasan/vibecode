/**
 * Main Authentication Service
 * Orchestrates all auth operations including login, registration, and OAuth
 */

import * as schema from '../schema';
import { eq, and, sql, or, lt, isNull } from 'drizzle-orm';
import { JWTUtils } from '../../utils/jwtUtils';
import { generateSecureToken } from '../../utils/cryptoUtils';
import { SessionService } from './SessionService';
import { PasswordService } from '../../utils/passwordService';
import { GoogleOAuthProvider } from '../../services/oauth/google';
import { GitHubOAuthProvider } from '../../services/oauth/github';
import { BaseOAuthProvider } from '../../services/oauth/base';
import { 
    SecurityError, 
    SecurityErrorType 
} from 'shared/types/errors';
import { AuthResult, AuthUserSession, OAuthUserInfo } from '../../types/auth-types';
import { generateId } from '../../utils/idGenerator';
import {
    AuthUser, 
    OAuthProvider
} from '../../types/auth-types';
import { mapUserResponse } from '../../utils/authUtils';
import { createLogger } from '../../logger';
import { validateEmail, validatePassword } from '../../utils/validationUtils';
import { extractRequestMetadata } from '../../utils/authUtils';
import { BaseService } from './BaseService';

const logger = createLogger('AuthService');

export interface LoginCredentials {
    email: string;
    password: string;
}

export interface RegistrationData {
    email: string;
    password: string;
    name?: string;
}

export class AuthService extends BaseService {
    private readonly sessionService: SessionService;
    private readonly passwordService: PasswordService;
    
    constructor(env: Env) {
        super(env);
        this.sessionService = new SessionService(env);
        this.passwordService = new PasswordService();
    }
    
    /**
     * Register a new user
     */
    async register(data: RegistrationData, request: Request): Promise<AuthResult> {
        try {
            const emailValidation = validateEmail(data.email);
            if (!emailValidation.valid) {
                throw new SecurityError(SecurityErrorType.INVALID_INPUT, emailValidation.error || 'Invalid email format', 400);
            }
            
            const passwordValidation = validatePassword(data.password, undefined, { email: data.email, name: data.name });
            if (!passwordValidation.valid) {
                throw new SecurityError(SecurityErrorType.INVALID_INPUT, passwordValidation.errors!.join(', '), 400);
            }
            
            const existingUser = await this.database.select().from(schema.users).where(eq(schema.users.email, data.email.toLowerCase())).get();
            if (existingUser) {
                throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Email already registered', 400);
            }
            
            const passwordHash = await this.passwordService.hash(data.password);
            const userId = generateId();
            const now = new Date();
            
            await this.database.insert(schema.users).values({
                id: userId,
                email: data.email.toLowerCase(),
                passwordHash,
                displayName: data.name || data.email.split('@')[0],
                emailVerified: true, 
                provider: 'email',
                providerId: userId,
                createdAt: now,
                updatedAt: now
            });
            
            const newUser = await this.database.select().from(schema.users).where(eq(schema.users.id, userId)).get();
            if (!newUser) {
                throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Failed to retrieve created user', 500);
            }
            
            await this.logAuthAttempt(data.email, 'register', true, request);
            
            const { accessToken, session } = await this.sessionService.createSession(userId, request);
            
            return {
                user: mapUserResponse(newUser),
                sessionId: session.sessionId,
                expiresAt: session.expiresAt,
                accessToken,
            };
        } catch (error) {
            await this.logAuthAttempt(data.email, 'register', false, request);
            if (error instanceof SecurityError) throw error;
            logger.error('Registration error', error);
            throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Registration failed', 500);
        }
    }
    
    /**
     * Login with email and password
     */
    async login(credentials: LoginCredentials, request: Request): Promise<AuthResult> {
        try {
            const user = await this.database.select().from(schema.users).where(and(eq(schema.users.email, credentials.email.toLowerCase()), sql`${schema.users.deletedAt} IS NULL`)).get();
            
            if (!user || !user.passwordHash) {
                await this.logAuthAttempt(credentials.email, 'login', false, request);
                throw new SecurityError(SecurityErrorType.UNAUTHORIZED, 'Invalid email or password', 401);
            }
            
            const passwordValid = await this.passwordService.verify(credentials.password, user.passwordHash);
            if (!passwordValid) {
                await this.logAuthAttempt(credentials.email, 'login', false, request);
                throw new SecurityError(SecurityErrorType.UNAUTHORIZED, 'Invalid email or password', 401);
            }
            
            const { accessToken, session } = await this.sessionService.createSession(user.id, request);
            await this.logAuthAttempt(credentials.email, 'login', true, request);
            
            return {
                user: mapUserResponse(user),
                accessToken,
                sessionId: session.sessionId,
                expiresAt: session.expiresAt,
            };
        } catch (error) {
            if (error instanceof SecurityError) throw error;
            logger.error('Login error', error);
            throw new SecurityError(SecurityErrorType.UNAUTHORIZED, 'Login failed', 500);
        }
    }
    
    async logout(sessionId: string): Promise<void> {
        try {
            await this.sessionService.revokeSessionId(sessionId);
        } catch (error) {
            logger.error('Logout error', error);
            throw new SecurityError(SecurityErrorType.UNAUTHORIZED, 'Logout failed', 500);
        }
    }

    async getOauthProvider(provider: OAuthProvider, request: Request): Promise<BaseOAuthProvider> {
        // PRODUCTION FIX: Prefer env.CUSTOM_DOMAIN to ensure callbacks match registered URLs
        // If not set, fall back to request origin (useful for local dev)
        let baseUrl = this.env.CUSTOM_DOMAIN 
            ? `https://${this.env.CUSTOM_DOMAIN}`
            : new URL(request.url).origin;

        // Strip trailing slash to avoid double slashes in callback URL
        if (baseUrl.endsWith('/')) {
            baseUrl = baseUrl.slice(0, -1);
        }
        
        switch (provider) {
            case 'google':
                return GoogleOAuthProvider.create(this.env, baseUrl);
            case 'github':
                return GitHubOAuthProvider.create(this.env, baseUrl);
            default:
                throw new SecurityError(SecurityErrorType.INVALID_INPUT, `OAuth provider ${provider} not configured`, 400);
        }
    }
    
    async getOAuthAuthorizationUrl(provider: OAuthProvider, request: Request, intendedRedirectUrl?: string): Promise<string> {
        const oauthProvider = await this.getOauthProvider(provider, request);
        
        await this.cleanupExpiredOAuthStates();
        
        // Validate intended redirect URL (post-login destination)
        let finalRedirectUrl: string | null = null;
        if (intendedRedirectUrl) {
            finalRedirectUrl = this.validateRedirectUrl(intendedRedirectUrl, request);
        }
        
        const state = generateSecureToken();
        const codeVerifier = BaseOAuthProvider.generateCodeVerifier();
        
        await this.database.insert(schema.oauthStates).values({
            id: generateId(),
            state,
            provider,
            codeVerifier,
            // CRITICAL FIX: Only store intended destination here. 
            // Do NOT fallback to provider.redirectUri (the callback URL), or you get a loop.
            redirectUri: finalRedirectUrl, 
            createdAt: new Date(),
            expiresAt: new Date(Date.now() + 600000), // 10 mins
            isUsed: false,
            scopes: [],
            userId: null,
            nonce: null
        });
        
        return await oauthProvider.getAuthorizationUrl(state, codeVerifier);
    }
    
    private async cleanupExpiredOAuthStates(): Promise<void> {
        try {
            await this.database.delete(schema.oauthStates)
                .where(or(lt(schema.oauthStates.expiresAt, new Date()), eq(schema.oauthStates.isUsed, true)));
        } catch (error) {
            logger.error('Error cleaning up OAuth states', error);
        }
    }
    
    async handleOAuthCallback(provider: OAuthProvider, code: string, state: string, request: Request): Promise<AuthResult> {
        try {
            const oauthProvider = await this.getOauthProvider(provider, request);
            
            const oauthState = await this.database.select().from(schema.oauthStates)
                .where(and(eq(schema.oauthStates.state, state), eq(schema.oauthStates.provider, provider), eq(schema.oauthStates.isUsed, false)))
                .get();
            
            if (!oauthState || new Date(oauthState.expiresAt) < new Date()) {
                throw new SecurityError(SecurityErrorType.CSRF_VIOLATION, 'Invalid or expired OAuth state', 400);
            }
            
            await this.database.update(schema.oauthStates).set({ isUsed: true }).where(eq(schema.oauthStates.id, oauthState.id));
            
            const tokens = await oauthProvider.exchangeCodeForTokens(code, oauthState.codeVerifier || undefined);
            const oauthUserInfo = await oauthProvider.getUserInfo(tokens.accessToken);
            const user = await this.findOrCreateOAuthUser(provider, oauthUserInfo);
            
            const { accessToken: sessionAccessToken, session } = await this.sessionService.createSession(user.id, request);
            await this.logAuthAttempt(user.email, `oauth_${provider}`, true, request);
            
            return {
                user: mapUserResponse(user),
                accessToken: sessionAccessToken,
                sessionId: session.sessionId,
                expiresAt: session.expiresAt,
                // Return null/undefined if no specific destination was requested
                redirectUrl: oauthState.redirectUri || undefined 
            };
        } catch (error) {
            await this.logAuthAttempt('', `oauth_${provider}`, false, request);
            if (error instanceof SecurityError) throw error;
            logger.error('OAuth callback error', error);
            throw new SecurityError(SecurityErrorType.UNAUTHORIZED, 'OAuth authentication failed', 500);
        }
    }
    
    private async findOrCreateOAuthUser(provider: OAuthProvider, oauthUserInfo: OAuthUserInfo): Promise<schema.User> {
        let user = await this.database.select().from(schema.users).where(eq(schema.users.email, oauthUserInfo.email.toLowerCase())).get();
        
        if (!user) {
            const userId = generateId();
            const now = new Date();
            await this.database.insert(schema.users).values({
                id: userId,
                email: oauthUserInfo.email.toLowerCase(),
                displayName: oauthUserInfo.name || oauthUserInfo.email.split('@')[0],
                avatarUrl: oauthUserInfo.picture,
                emailVerified: oauthUserInfo.emailVerified || false,
                provider: provider,
                providerId: oauthUserInfo.id,
                createdAt: now,
                updatedAt: now
            });
            user = await this.database.select().from(schema.users).where(eq(schema.users.id, userId)).get();
        } else {
            // Update existing user with latest OAuth data
            await this.database.update(schema.users).set({
                displayName: oauthUserInfo.name || user.displayName,
                avatarUrl: oauthUserInfo.picture || user.avatarUrl,
                provider: provider,
                providerId: oauthUserInfo.id,
                emailVerified: oauthUserInfo.emailVerified || user.emailVerified,
                updatedAt: new Date()
            }).where(eq(schema.users.id, user.id));
            user = await this.database.select().from(schema.users).where(eq(schema.users.id, user.id)).get();
        }
        return user!;
    }
    
    private async logAuthAttempt(identifier: string, attemptType: string, success: boolean, request: Request): Promise<void> {
        try {
            const requestMetadata = extractRequestMetadata(request);
            await this.database.insert(schema.authAttempts).values({
                identifier: identifier.toLowerCase(),
                attemptType: attemptType as any,
                success: success,
                ipAddress: requestMetadata.ipAddress
            });
        } catch (error) {
            logger.error('Failed to log auth attempt', error);
        }
    }
    
    private validateRedirectUrl(redirectUrl: string, request: Request): string | null {
        try {
            const requestUrl = new URL(request.url);
            const redirectUrlObj = redirectUrl.startsWith('/') ? new URL(redirectUrl, requestUrl.origin) : new URL(redirectUrl);
            
            // Allow redirects to same origin OR specific allowed frontend hosts if needed (configure as env var in future)
            if (redirectUrlObj.origin !== requestUrl.origin) {
                // If using custom domain, ensure redirect matches custom domain
                if (this.env.CUSTOM_DOMAIN && !redirectUrlObj.hostname.endsWith(this.env.CUSTOM_DOMAIN)) {
                     logger.warn('OAuth redirect URL rejected', { redirectUrl });
                     return null;
                }
            }
            
            const authPaths = ['/api/auth/', '/logout'];
            if (authPaths.some(path => redirectUrlObj.pathname.startsWith(path))) return null;
            
            return redirectUrl;
        } catch (error) {
            return null;
        }
    }

    // ... Verification methods (kept same as original) ...
    private async generateAndStoreVerificationOtp(email: string): Promise<void> {
        const otp = Math.floor(100000 + Math.random() * 900000).toString();
        const expiresAt = new Date(Date.now() + 15 * 60 * 1000);
        await this.database.insert(schema.verificationOtps).values({
            id: generateId(),
            email: email.toLowerCase(),
            otp: await this.passwordService.hash(otp),
            expiresAt,
            createdAt: new Date()
        });
        logger.info('Verification OTP generated', { email, otp: otp.slice(0, 2) + '****' });
    }

    async verifyEmailWithOtp(email: string, otp: string, request: Request): Promise<AuthResult> {
        try {
            const storedOtp = await this.database.select().from(schema.verificationOtps)
                .where(and(eq(schema.verificationOtps.email, email.toLowerCase()), eq(schema.verificationOtps.used, false), sql`${schema.verificationOtps.expiresAt} > ${new Date()}`))
                .orderBy(sql`${schema.verificationOtps.createdAt} DESC`).get();

            if (!storedOtp) throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Invalid or expired verification code', 400);

            const otpValid = await this.passwordService.verify(otp, storedOtp.otp);
            if (!otpValid) throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Invalid verification code', 400);

            await this.database.update(schema.verificationOtps).set({ used: true, usedAt: new Date() }).where(eq(schema.verificationOtps.id, storedOtp.id));

            const user = await this.database.select().from(schema.users).where(eq(schema.users.email, email.toLowerCase())).get();
            if (!user) throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'User not found', 404);

            await this.database.update(schema.users).set({ emailVerified: true, updatedAt: new Date() }).where(eq(schema.users.id, user.id));

            const { accessToken, session } = await this.sessionService.createSession(user.id, request);
            await this.logAuthAttempt(email, 'email_verification', true, request);

            return {
                user: mapUserResponse({ ...user, emailVerified: true }),
                accessToken,
                sessionId: session.sessionId,
                expiresAt: session.expiresAt,
            };
        } catch (error) {
            await this.logAuthAttempt(email, 'email_verification', false, request);
            if (error instanceof SecurityError) throw error;
            throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Email verification failed', 500);
        }
    }

    async getUserForAuth(userId: string): Promise<AuthUser | null> {
        try {
            const user = await this.database.select({
                id: schema.users.id,
                email: schema.users.email,
                displayName: schema.users.displayName,
                username: schema.users.username,
                avatarUrl: schema.users.avatarUrl,
                bio: schema.users.bio,
                timezone: schema.users.timezone,
                provider: schema.users.provider,
                emailVerified: schema.users.emailVerified,
                createdAt: schema.users.createdAt,
            }).from(schema.users).where(and(eq(schema.users.id, userId), isNull(schema.users.deletedAt))).get();
            
            return user ? mapUserResponse(user) : null;
        } catch (error) {
            return null;
        }
    }
    
    async validateTokenAndGetUser(token: string, env: Env): Promise<AuthUserSession | null> {
        try {
            const jwtUtils = JWTUtils.getInstance(env);
            const payload = await jwtUtils.verifyToken(token);
            if (!payload || payload.type !== 'access' || payload.exp * 1000 < Date.now()) return null;
            
            const user = await this.getUserForAuth(payload.sub);
            return user ? { user, sessionId: payload.sessionId } : null;
        } catch (error) {
            return null;
        }
    }
    
    async resendVerificationOtp(email: string): Promise<void> {
        try {
            const user = await this.database.select().from(schema.users).where(eq(schema.users.email, email.toLowerCase())).get();
            if (!user) throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'No account found', 404);
            if (user.emailVerified) throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Already verified', 400);

            await this.database.update(schema.verificationOtps).set({ used: true, usedAt: new Date() })
                .where(and(eq(schema.verificationOtps.email, email.toLowerCase()), eq(schema.verificationOtps.used, false)));

            await this.generateAndStoreVerificationOtp(email.toLowerCase());
        } catch (error) {
            if (error instanceof SecurityError) throw error;
            throw new SecurityError(SecurityErrorType.INVALID_INPUT, 'Failed to resend verification code', 500);
        }
    }
}
