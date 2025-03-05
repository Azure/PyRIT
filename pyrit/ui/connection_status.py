# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gradio as gr

from rpc_client import RPCClient

class ConnectionStatusHandler:
    def __init__(self,
                 is_connected_state: gr.State,
                 rpc_client: RPCClient):
        self.state = is_connected_state
        self.server_disconnected = False
        self.rpc_client = rpc_client
        self.converted_value = ""
    
    def setup(self, *, main_interface: gr.Column, loading_animation: gr.Column, converted_value_state: gr.State):
        self.state.change(fn=self._on_state_change, inputs=[self.state], outputs=[main_interface, loading_animation, converted_value_state])

        connection_status_timer = gr.Timer(1)
        connection_status_timer.tick(
            fn=self._check_connection_status,
            inputs=[self.state],
            outputs=[self.state]
        ).then(
            fn=self._reconnect_if_needed,
            outputs=[self.state]
        )

    def set_ready(self):
        self.server_disconnected = False

    def set_disconnected(self):
        self.server_disconnected = True

    def set_converted_value(self, converted_value: str):
        self.converted_value = converted_value

    def _on_state_change(self, is_connected: bool):
        print("Connection status changed to: ", is_connected, " - ", self.converted_value)
        if is_connected:
            return [gr.Column(visible=True), gr.Row(visible=False), self.converted_value]
        return [gr.Column(visible=False), gr.Row(visible=True), self.converted_value]

    def _check_connection_status(self, is_connected: bool):
        if self.server_disconnected or not is_connected:
            print("Gradio disconnected")
            return False
        return True
    
    def _reconnect_if_needed(self):
        if self.server_disconnected:
            print("Attempting to reconnect")
            self.rpc_client.reconnect()
            prompt = self.rpc_client.wait_for_prompt()
            self.converted_value = str(converted_value.original_value)
            self.server_disconnected = False
        return True