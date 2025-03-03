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
        self.next_prompt = ""
    
    def setup(self, main_interface: gr.Column, loading_animation: gr.Column, next_prompt_state: gr.State):
        self.state.change(fn=self.__on_state_change, inputs=[self.state], outputs=[main_interface, loading_animation, next_prompt_state])

        connection_status_timer = gr.Timer(1)
        connection_status_timer.tick(
            fn=self.__check_connection_status,
            inputs=[self.state],
            outputs=[self.state]
        ).then(
            fn=self.__reconnect_if_needed,
            outputs=[self.state]
        )

    def set_ready(self):
        self.server_disconnected = False

    def set_disconnected(self):
        self.server_disconnected = True

    def set_next_prompt(self, next_prompt: str):
        self.next_prompt = next_prompt

    def __on_state_change(self, is_connected: bool):
        print("Connection status changed to: ", is_connected, " - ", self.next_prompt)
        if is_connected:
            return [gr.Column(visible=True), gr.Row(visible=False), self.next_prompt]
        return [gr.Column(visible=False), gr.Row(visible=True), self.next_prompt]

    def __check_connection_status(self, is_connected: bool):
        if self.server_disconnected or not is_connected:
            print("Gradio disconnected")
            return False
        return True
    
    def __reconnect_if_needed(self):
        if self.server_disconnected:
            print("Attempting to reconnect")
            self.rpc_client.reconnect()
            prompt = self.rpc_client.wait_for_prompt()
            self.next_prompt = str(prompt.original_value)
            self.server_disconnected = False
        return True