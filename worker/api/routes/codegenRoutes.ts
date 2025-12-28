import { CodingAgentController } from '../controllers/agent/controller';
import { AppEnv } from '../../types/appenv';
import { Hono } from 'hono';
import { AuthConfig, setAuthLevel } from '../../middleware/auth/routeAuth';
import { adaptController } from '../honoAdapter';

/**
 * Setup and configure the application router
 */
export function setupCodegenRoutes(app: Hono<AppEnv>): void {
    // ========================================
    // CODE GENERATION ROUTES
    // ========================================
    
    // Create new app - requires authentication
    app.post(
        '/api/agent', 
        setAuthLevel(AuthConfig.authenticated), 
        adaptController(CodingAgentController, CodingAgentController.startCodeGeneration)
    );
    
    // ========================================
    // APP EDITING ROUTES (/chat/:id)
    // ========================================
    
    // WebSocket for app editing - OWNER ONLY
    app.get(
        '/api/agent/:agentId/ws', 
        setAuthLevel(AuthConfig.ownerOnly), 
        adaptController(CodingAgentController, CodingAgentController.handleWebSocketConnection)
    );
    
    // Connect to existing agent for editing - OWNER ONLY
    app.get(
        '/api/agent/:agentId/connect', 
        setAuthLevel(AuthConfig.ownerOnly), 
        adaptController(CodingAgentController, CodingAgentController.connectToExistingAgent)
    );

    // Deployment preview - AUTHENTICATED
    app.get(
        '/api/agent/:agentId/preview', 
        setAuthLevel(AuthConfig.authenticated), 
        adaptController(CodingAgentController, CodingAgentController.deployPreview)
    );
}
