# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import gradio as gr
import webview
from connection_status import ConnectionStatusHandler
from rpc_client import RPCClient

GRADIO_POLLING_RATE = 0.5  # Polling Rate by the Gradio UI


class GradioApp:
    def __init__(self):
        self.i = 0
        self.rpc_client = RPCClient(self._disconnected_rpc_callback)
        self.connect_status = None
        self.url = ""

    def start_gradio(self, open_browser=False):
        with gr.Blocks() as demo:
            is_connected = gr.State(False)
            next_prompt_state = gr.State("")

            self.connect_status = ConnectionStatusHandler(is_connected, self.rpc_client)
            with gr.Column(visible=False) as main_interface:
                prompt = gr.Markdown("Prompt: ")
                prompt.height = "200px"
                with gr.Row():
                    safe = gr.Button("Safe")
                    unsafe = gr.Button("Unsafe")

                    safe.click(
                        fn=lambda: [gr.update(interactive=False)] * 2 + [""], outputs=[safe, unsafe, next_prompt_state]
                    ).then(fn=self._safe_clicked, outputs=next_prompt_state)
                    unsafe.click(
                        fn=lambda: [gr.update(interactive=False)] * 2 + [""], outputs=[safe, unsafe, next_prompt_state]
                    ).then(fn=self._unsafe_clicked, outputs=next_prompt_state)

            with gr.Row() as loading_animation:
                loading_text = gr.Markdown("Connecting to PyRIT")
                timer = gr.Timer(GRADIO_POLLING_RATE)
                timer.tick(fn=self._loading_dots, outputs=loading_text)

            next_prompt_state.change(
                fn=self._on_next_prompt_change, inputs=[next_prompt_state], outputs=[prompt, safe, unsafe]
            )
            self.connect_status.setup(
                main_interface=main_interface, loading_animation=loading_animation, next_prompt_state=next_prompt_state
            )

            demo.load(
                fn=self._main_interface_loaded,
                outputs=[main_interface, loading_animation, next_prompt_state, is_connected],
            )

        if open_browser:
            demo.launch(inbrowser=True)
        else:
            _, url, _ = demo.launch(prevent_thread_lock=True)
            self.url = url
            print("Gradio launched")
            webview.create_window("PyRIT - Scorer", self.url)
            webview.start()
            print("Webview closed!")

        if self.rpc_client:
            self.rpc_client.stop()

    def _safe_clicked(self):
        return self._send_prompt_response(True)

    def _unsafe_clicked(self):
        return self._send_prompt_response(False)

    def _send_prompt_response(self, value):
        self.rpc_client.send_prompt_response(value)
        prompt_request = self.rpc_client.wait_for_prompt()
        return str(prompt_request.converted_value)

    def _on_next_prompt_change(self, next_prompt):
        if next_prompt == "":
            return [
                gr.Markdown("Waiting for next prompt..."),
                gr.update(interactive=False),
                gr.update(interactive=False),
            ]
        return [gr.Markdown("Prompt: " + next_prompt), gr.update(interactive=True), gr.update(interactive=True)]

    def _loading_dots(self):
        self.i = (self.i + 1) % 4
        return gr.Markdown("Connecting to PyRIT" + "." * self.i)

    def _disconnected_rpc_callback(self):
        self.connect_status.set_disconnected()

    def _main_interface_loaded(self):
        print("Showing main interface")
        self.rpc_client.start()
        prompt_request = self.rpc_client.wait_for_prompt()
        next_prompt = str(prompt_request.converted_value)
        self.connect_status.set_next_prompt(next_prompt)
        self.connect_status.set_ready()
        print("PyRIT connected")
        return [gr.Column(visible=True), gr.Row(visible=False), next_prompt, True]
