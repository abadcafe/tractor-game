import type { ServerMessage, ClientAction } from "../core/types.ts";
import { WS_PATH } from "../config.ts";

/**
 * WebSocket client that manages a single connection to the game server.
 * Supports automatic reconnection with exponential backoff.
 */
export class WsClient {
  private _ws: WebSocket | null = null;
  private _messageHandler: ((msg: ServerMessage) => void) | null = null;
  private _disconnectHandler: (() => void) | null = null;
  private _reconnectFailHandler: (() => void) | null = null;
  private _wsHost = "";
  private _reconnectAttempts = 0;
  private _reconnectGameId = "";
  private _reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _disconnecting = false;
  private _reconnecting = false;

  /** Register a handler for incoming messages. Must be called before connect(). */
  onMessage(handler: (msg: ServerMessage) => void): void {
    this._messageHandler = handler;
  }

  /** Register a handler for disconnection events. */
  onDisconnect(handler: () => void): void {
    this._disconnectHandler = handler;
  }

  /** Register a handler for when all reconnection attempts fail. */
  onReconnectFail(handler: () => void): void {
    this._reconnectFailHandler = handler;
  }

  /** Set the WebSocket host URL (e.g. "ws://localhost:8080"). */
  setWsHost(host: string): void {
    this._wsHost = host;
  }

  /** Connect to the game server. Constructs the WebSocket URL from gameId and wsHost. */
  connect(gameId: string, wsHost?: string): Promise<void> {
    if (wsHost !== undefined) {
      this._wsHost = wsHost;
    }
    if (!this._wsHost) {
      return Promise.reject(new Error("WebSocket host not set. Call setWsHost() or provide wsHost parameter."));
    }
    this._disconnecting = false;
    this._reconnectGameId = gameId;
    this._reconnectAttempts = 0;

    return this._doConnect(gameId, this._wsHost);
  }

  /** Send an action to the server. */
  send(action: ClientAction): void {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(action));
    } else {
      console.warn("[WsClient] send() called but socket not open, action discarded:", action);
    }
  }

  /** Disconnect from the server and cancel any pending reconnection. */
  disconnect(): void {
    this._disconnecting = true;
    this._reconnecting = false;
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    this._reconnectAttempts = 0;
  }

  /** Return true if currently attempting to reconnect. */
  get isReconnecting(): boolean {
    return this._reconnecting;
  }

  private _doConnect(gameId: string, wsHost: string): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const url = `${wsHost}${WS_PATH(gameId)}`;
      const ws = new WebSocket(url);
      this._ws = ws;
      let settled = false;

      ws.addEventListener("open", () => {
        if (!settled) {
          settled = true;
          this._reconnectAttempts = 0;
          this._reconnecting = false;
          resolve();
        }
      });

      ws.addEventListener("message", (event) => {
        let msg: ServerMessage;
        try {
          msg = JSON.parse(event.data as string);
        } catch {
          console.warn("[WsClient] Malformed message ignored:", event.data);
          return;
        }
        this._messageHandler?.(msg);
      });

      ws.addEventListener("close", () => {
        if (!settled) {
          settled = true;
          reject(new Error("WebSocket connection failed"));
        }
        this._ws = null;
        if (!this._disconnecting) {
          this._disconnectHandler?.();
          this._attemptReconnect();
        }
      });

      ws.addEventListener("error", () => {
        // On error, the close event will fire next which handles
        // promise settlement and reconnection.
      });
    });
  }

  private _attemptReconnect(): void {
    if (this._disconnecting) {
      return;
    }

    if (this._reconnectAttempts >= 3) {
      // All reconnection attempts failed — notify the user
      this._reconnecting = false;
      console.error("[WsClient] All reconnection attempts failed. Please refresh the page.");
      this._reconnectFailHandler?.();
      return;
    }

    this._reconnecting = true;
    const delay = 1000 * Math.pow(2, this._reconnectAttempts);
    this._reconnectAttempts++;

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      this._doConnect(this._reconnectGameId, this._wsHost).catch(() => {
        // Reconnection failure is already handled by the close event
        // listener in _doConnect, which triggers the next reconnect attempt.
      });
    }, delay);
  }
}
