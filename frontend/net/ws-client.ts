import type { ClientAction, ServerMessage } from "../core/protocol.ts";
import { WS_PATH } from "../config.ts";

/**
 * WebSocket client that manages a single connection to the game server.
 * Supports automatic reconnection with exponential backoff.
 * Tracks seq from server for the action protocol.
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
  private _connectGeneration = 0;

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

  /** Connect to the game server. Constructs the WebSocket URL from gameId and wsHost. */
  connect(gameId: string, wsHost?: string): Promise<void> {
    if (wsHost !== undefined) {
      this._wsHost = wsHost;
    }
    if (!this._wsHost) {
      return Promise.reject(
        new Error(
          "WebSocket host not set. Call setWsHost() or provide wsHost parameter.",
        ),
      );
    }
    this._connectGeneration++;
    this._cancelReconnectTimer();
    this._disconnecting = false;
    this._reconnectGameId = gameId;
    this._reconnectAttempts = 0;

    return this._doConnect(
      gameId,
      this._wsHost,
      this._connectGeneration,
    );
  }

  /** Send an action to the server. The seq is already included in the ClientAction. */
  send(action: ClientAction): boolean {
    if (this._ws && this._ws.readyState === WebSocket.OPEN) {
      this._ws.send(JSON.stringify(action));
      return true;
    }
    console.warn(
      "[WsClient] send() called but socket not open, action discarded:",
      action,
    );
    return false;
  }

  /** Request the current state when the client does not know the latest seq. */
  requestState(): boolean {
    return this.send({ seq: 0 });
  }

  /** Disconnect from the server and cancel any pending reconnection. */
  disconnect(): void {
    this._connectGeneration++;
    this._disconnecting = true;
    this._reconnecting = false;
    this._cancelReconnectTimer();
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

  /** Return true only while the current socket is open and usable. */
  get isConnected(): boolean {
    return this._ws?.readyState === WebSocket.OPEN;
  }

  private _doConnect(
    gameId: string,
    wsHost: string,
    generation: number,
  ): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const url = `${wsHost}${WS_PATH(gameId)}`;
      const ws = new WebSocket(url);
      this._ws = ws;
      let settled = false;
      let opened = false;

      ws.addEventListener("open", () => {
        if (!this._isCurrentSocket(ws, generation)) {
          ws.close();
          return;
        }
        opened = true;
        this.requestState();
        if (!settled) {
          settled = true;
          this._reconnectAttempts = 0;
          this._reconnecting = false;
          resolve();
        }
      });

      ws.addEventListener("message", (event) => {
        if (!this._isCurrentSocket(ws, generation)) {
          return;
        }
        let msg: ServerMessage;
        try {
          msg = JSON.parse(event.data as string);
        } catch {
          console.warn(
            "[WsClient] Malformed message ignored:",
            event.data,
          );
          return;
        }
        this._messageHandler?.(msg);
      });

      ws.addEventListener("close", () => {
        const isCurrent = this._isCurrentSocket(ws, generation);
        if (!settled) {
          settled = true;
          reject(new Error("WebSocket connection failed"));
        }
        if (!isCurrent) {
          return;
        }
        this._ws = null;
        if (!this._disconnecting && opened) {
          this._disconnectHandler?.();
          this._attemptReconnect(generation);
        }
      });

      ws.addEventListener("error", () => {
        // On error, the close event will fire next which handles
        // promise settlement and reconnection.
      });
    });
  }

  private _attemptReconnect(generation: number): void {
    if (this._disconnecting || generation !== this._connectGeneration) {
      return;
    }

    if (this._reconnectAttempts >= 3) {
      // All reconnection attempts failed — notify the user
      this._reconnecting = false;
      console.error(
        "[WsClient] All reconnection attempts failed. Please refresh the page.",
      );
      this._reconnectFailHandler?.();
      return;
    }

    this._reconnecting = true;
    const delay = 1000 * Math.pow(2, this._reconnectAttempts);
    this._reconnectAttempts++;

    this._reconnectTimer = setTimeout(() => {
      this._reconnectTimer = null;
      if (
        generation !== this._connectGeneration || this._disconnecting
      ) {
        return;
      }
      this._doConnect(this._reconnectGameId, this._wsHost, generation)
        .catch(() => {
          // Reconnection failure is already handled by the close event
          // listener in _doConnect, which triggers the next reconnect attempt.
        });
    }, delay);
  }

  private _cancelReconnectTimer(): void {
    if (this._reconnectTimer !== null) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    this._reconnecting = false;
  }

  private _isCurrentSocket(ws: WebSocket, generation: number): boolean {
    return generation === this._connectGeneration && this._ws === ws;
  }
}
