# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import rpyc
import time
import socket

from typing import Callable, Optional
from threading import Thread, Semaphore, Event
from pyrit.models import PromptRequestPiece, Score

DEFAULT_PORT = 18812

class RPCClient:
    def __init__(self, callback_disconnected: Optional[Callable] = None):
        self._c = None
        self._bgsrv = None
        
        self._ping_thread = None
        self._bgsrv_thread = None
        self._is_running = False

        self._shutdown_event = None
        self._prompt_received_sem = None

        self._prompt_received = None
        self._callback_disconnected = callback_disconnected

    def start(self):
        # Check if the port is open
        self._wait_for_server_avaible()
        self._prompt_received_sem = Semaphore(0)        
        
        self._c = rpyc.connect("localhost", DEFAULT_PORT, config={'allow_public_attrs': True})
        self._is_running = True
        self._shutdown_event = Event()
        self._bgsrv_thread = Thread(target=self._bgsrv_lifecycle)
        self._bgsrv_thread.start()

    def wait_for_prompt(self) -> PromptRequestPiece:
        self._prompt_received_sem.acquire()
        return self._prompt_received
    
    def send_prompt_response(self, response: bool):
        score = Score(
            score_value=str(response),
            score_type="true_false",
            score_category="safety",
            score_value_description="Safe" if response else "Unsafe",
            score_rationale="The prompt is safe" if response else "The prompt is unsafe",
            score_metadata={"prompt_target_identifier": self._prompt_received.prompt_target_identifier},
            prompt_request_response_id=self._prompt_received.conversation_id
        )
        self._c.root.receive_score(score)

    def _wait_for_server_avaible(self):
        # Wait for the server to be available
        while not self._is_server_running():
            print("Server is not running. Waiting for server to start...")
            time.sleep(1)

    def stop(self):
        """
        Stop the client.
        """
        # Send a signal to the thread to stop
        self._shutdown_event.set()
    
    def reconnect(self):
        """
        Reconnect to the server.
        """
        self.stop()
        print("Reconnecting to server...")
        self.start()

    def _receive_prompt(self, prompt_request: PromptRequestPiece, task: Optional[str] = None):
        print(f"Received prompt: {prompt_request}")
        self._prompt_received = prompt_request
        self._prompt_received_sem.release()

    def _ping(self):
        try:
            while self._is_running:
                self._c.root.receive_ping()
                time.sleep(1.5)
        except EOFError:
            print("Connection closed")
            if self._callback_disconnected is not None:
                self._callback_disconnected()

    def _bgsrv_lifecycle(self):
        self._bgsrv = rpyc.BgServingThread(self._c)
        self._ping_thread = Thread(target=self._ping)
        self._ping_thread.start()
        
        # Register callback
        self._c.root.callback_score_prompt(self._receive_prompt)

        # Wait for the server to be disconnected
        self._shutdown_event.wait()

        self._is_running = False
        self._ping_thread.join()
        
        # Avoid calling stop() twice if the server is already stopped. This can happen if the server is stopped
        # by the ping request.
        if self._bgsrv._active:
            self._bgsrv.stop()
    
    def _is_server_running(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', DEFAULT_PORT)) == 0