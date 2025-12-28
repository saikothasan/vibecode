/**
 * Centralized Authentication Utilities
 */

import type { AuthUser } from '../types/auth-types';
import type { User } from '../database/schema';

export function extractSessionId(request: Request): string | null {
    const cookieHeader = request.headers.get('Cookie');
    if (!cookieHeader) return null;
    const cookies = parseCookies(cookieHeader);
    return cookies['sessionId'];
}

export enum TokenExtractionMethod {
	AUTHORIZATION_HEADER = 'authorization_header',
	COOKIE = 'cookie',
	QUERY_PARAMETER = 'query_parameter',
}

export interface TokenExtractionResult {
	token: string | null;
	method?: TokenExtractionMethod;
	cookieName?: string;
}

export function extractToken(request: Request): string | null {
	const result = extractTokenWithMetadata(request);
	return result.token;
}

export function extractTokenWithMetadata(request: Request): TokenExtractionResult {
	const authHeader = request.headers.get('Authorization');
    // Case insensitive regex for Bearer
	if (authHeader && /^[Bb]earer /.test(authHeader)) {
		const token = authHeader.substring(7).trim();
		if (token.length > 0) {
			return { token, method: TokenExtractionMethod.AUTHORIZATION_HEADER };
		}
	}

	const cookieHeader = request.headers.get('Cookie');
	if (cookieHeader) {
		const cookies = parseCookies(cookieHeader);
		const cookieNames = ['accessToken', 'auth_token', 'jwt'];
		for (const cookieName of cookieNames) {
			if (cookies[cookieName]) {
				return { token: cookies[cookieName], method: TokenExtractionMethod.COOKIE, cookieName };
			}
		}
	}

	const url = new URL(request.url);
	const queryToken = url.searchParams.get('token') || url.searchParams.get('access_token');
	if (queryToken && queryToken.length > 0) {
		return { token: queryToken, method: TokenExtractionMethod.QUERY_PARAMETER };
	}

	return { token: null };
}

export function parseCookies(cookieHeader: string): Record<string, string> {
	const cookies: Record<string, string> = {};
	const pairs = cookieHeader.split(';');
	for (const pair of pairs) {
		const [key, value] = pair.trim().split('=');
		if (key && value) {
			cookies[key] = decodeURIComponent(value);
		}
	}
	return cookies;
}

export function clearAuthCookie(name: string): string {
	return createSecureCookie({ name, value: '', maxAge: 0 });
}

export function clearAuthCookies(response: Response): void {
	response.headers.append('Set-Cookie', clearAuthCookie('accessToken'));
	response.headers.append('Set-Cookie', clearAuthCookie('auth_token'));
    response.headers.append('Set-Cookie', clearAuthCookie('sessionId'));
}

export interface CookieOptions {
	name: string;
	value: string;
	maxAge?: number; // seconds
	httpOnly?: boolean;
	secure?: boolean;
	sameSite?: 'Strict' | 'Lax' | 'None';
	path?: string;
	domain?: string;
}

export function createSecureCookie(options: CookieOptions): string {
	const {
		name,
		value,
		maxAge = 7 * 24 * 60 * 60,
		secure = true, // Default to secure
		sameSite = 'Lax',
		path = '/',
		domain,
        httpOnly = true
	} = options;

	const parts = [`${name}=${encodeURIComponent(value)}`];

	if (maxAge >= 0) parts.push(`Max-Age=${maxAge}`);
	if (path) parts.push(`Path=${path}`);
	if (domain) parts.push(`Domain=${domain}`);
	if (secure) parts.push('Secure');
	if (sameSite) parts.push(`SameSite=${sameSite}`);
    if (httpOnly) parts.push('HttpOnly');

	return parts.join('; ');
}

export function setSecureAuthCookies(
	response: Response,
	tokens: {
		accessToken: string;
		accessTokenExpiry?: number;
	},
): void {
	const {
		accessToken,
		accessTokenExpiry = 3 * 24 * 60 * 60,
	} = tokens;

	response.headers.append(
		'Set-Cookie',
		createSecureCookie({
			name: 'accessToken',
			value: accessToken,
			maxAge: accessTokenExpiry,
			sameSite: 'Lax',
            // In dev (localhost), secure cookies might fail if not on HTTPS. 
            // Ideally, this should be configurable, but defaulting to true is safer.
			secure: true, 
		}),
	);
}

export interface RequestMetadata {
	ipAddress: string;
	userAgent: string;
	referer?: string;
	origin?: string;
	acceptLanguage?: string;
	cfConnectingIp?: string;
	cfRay?: string;
	cfCountry?: string;
	cfTimezone?: string;
}

export function extractRequestMetadata(request: Request): RequestMetadata {
	const headers = request.headers;
	return {
		ipAddress:
			headers.get('CF-Connecting-IP') ||
			headers.get('X-Forwarded-For')?.split(',')[0]?.trim() ||
			headers.get('X-Real-IP') ||
			'unknown',
		userAgent: headers.get('User-Agent') || 'unknown',
		referer: headers.get('Referer') || undefined,
		origin: headers.get('Origin') || undefined,
		acceptLanguage: headers.get('Accept-Language') || undefined,
		cfConnectingIp: headers.get('CF-Connecting-IP') || undefined,
		cfRay: headers.get('CF-Ray') || undefined,
		cfCountry: headers.get('CF-IPCountry') || undefined,
		cfTimezone: headers.get('CF-Timezone') || undefined,
	};
}

export interface SessionResponse {
	user: AuthUser;
    sessionId: string;
    expiresAt: Date | null;
}

export function mapUserResponse(
	user: (Partial<User> & { id: string; email: string }) | AuthUser,
): AuthUser {
	if ('isAnonymous' in user) {
		return user as AuthUser;
	}
	return {
		id: user.id,
		email: user.email,
		displayName: user.displayName || undefined,
		username: user.username || undefined,
		avatarUrl: user.avatarUrl || undefined,
		bio: user.bio || undefined,
		timezone: user.timezone || undefined,
		provider: user.provider || undefined,
		emailVerified: user.emailVerified || undefined,
		createdAt: user.createdAt || undefined,
	};
}

export function formatAuthResponse(
	user: AuthUser,
	sessionId: string,
	expiresAt: Date | null,
): SessionResponse {
	return { user, sessionId, expiresAt };
}
