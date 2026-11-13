/**
 * VisionPort WebSocket Client
 * Real-time telemetry, live detection event stream, and video frames
 */

class VisionPortWebSocket {
  constructor(endpoint = "/ws/monitoring") {
    this.endpoint = endpoint;
    this.socket = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelayMs = 2500;
    this.listeners = {
      telemetry: [],
      connect: [],
      disconnect: [],
      error: [],
    };
  }

  getWebSocketUrl() {
    const loc = window.location;
    const proto = loc.protocol === "https:" ? "wss:" : "ws:";
    const host = loc.host || "127.0.0.1:8000";
    return `${proto}//${host}${this.endpoint}`;
  }

  connect() {
    if (this.socket && (this.socket.readyState === WebSocket.OPEN || this.socket.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const url = this.getWebSocketUrl();
    try {
      this.socket = new WebSocket(url);
    } catch (err) {
      this._emit("error", err);
      this._scheduleReconnect();
      return;
    }

    this.socket.onopen = () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      this._emit("connect", { url });
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "telemetry") {
          this._emit("telemetry", data);
        }
      } catch (err) {
        console.warn("[WS] Failed to parse message:", event.data);
      }
    };

    this.socket.onclose = (event) => {
      this.isConnected = false;
      this._emit("disconnect", event);
      this._scheduleReconnect();
    };

    this.socket.onerror = (err) => {
      this._emit("error", err);
    };
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.isConnected = false;
  }

  _scheduleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        if (!this.isConnected) {
          this.connect();
        }
      }, this.reconnectDelayMs);
    }
  }

  on(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event].push(callback);
    }
  }

  off(event, callback) {
    if (this.listeners[event]) {
      this.listeners[event] = this.listeners[event].filter((cb) => cb !== callback);
    }
  }

  _emit(event, data) {
    if (this.listeners[event]) {
      this.listeners[event].forEach((cb) => {
        try {
          cb(data);
        } catch (e) {
          console.error(`[WS] Error in ${event} listener:`, e);
        }
      });
    }
  }
}

// Global WebSocket manager instance
const wsClient = new VisionPortWebSocket("/ws/monitoring");

